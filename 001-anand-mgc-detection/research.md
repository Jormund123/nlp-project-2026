# Research & Technology Decisions: MGC Detection

**Date**: 2026-01-22  
**Feature**: Machine-Generated Code Detection (Anand's Tasks)  
**Purpose**: Document technology choices and implementation patterns

---

## 1. Model Selection: CodeBERT

### Decision: Use `microsoft/codebert-base` with truncation strategy

**Rationale**:

- **Pre-trained on code**: CodeBERT is specifically trained on code from GitHub (6 languages)
- **Bi-modal**: Understands both natural language (comments) and code
- **Proven**: Strong performance on code understanding tasks (code search, clone detection)
- **Compatible**: Standard transformers architecture, easy to fine-tune

**Context Length Challenge (512 tokens)**:

CodeBERT has a 512-token limit. AI-generated boilerplate often appears at the beginning or end of files. For files exceeding 512 tokens, we need a truncation strategy:

| Strategy                 | Pros                  | Cons                        | Chosen?      |
| ------------------------ | --------------------- | --------------------------- | ------------ |
| First 512                | Simple, fast          | Misses end-of-file patterns | ✅ Default   |
| Last 512                 | Captures end patterns | Misses imports/headers      |              |
| Middle 512               | Captures core logic   | Misses both ends            |              |
| Sliding window + pooling | Full coverage         | 3-4x slower training        | For ablation |

**Recommendation**: Start with first 512 tokens (most AI patterns appear in function headers/docstrings). If time permits, run an ablation comparing truncation strategies.

**Alternatives Considered**:

1. **GraphCodeBERT**
   - Pros: Uses data flow graphs, potentially better semantic understanding
   - Cons: Requires language-specific parsers for DFG extraction, complex setup
   - Rejected because: Time-consuming implementation, hypothesis is weak for this task

2. **StarCoder2 (Long Context)**
   - Pros: Handles longer code, newer model
   - Cons: _Generative_ model, not encoder. Would require perplexity-based approach.
   - Rejected because: Completely different paradigm, doesn't fit our classification approach

3. **BERT/RoBERTa**
   - Pros: Well-documented, many examples
   - Cons: Not code-specific, likely worse performance
   - Rejected because: CodeBERT designed specifically for code

**Implementation**:

```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModel.from_pretrained("microsoft/codebert-base")

# Add classification head
class CodeBERTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.codebert = AutoModel.from_pretrained("microsoft/codebert-base")
        self.classifier = nn.Linear(768, 2)  # binary classification

    def forward(self, input_ids, attention_mask):
        outputs = self.codebert(input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0]  # [CLS] token
        return self.classifier(cls_output)
```

---

## 2. TF-IDF Configuration

### Decision: max_features=10000, ngram_range=(1, 3) with word tokenization

**Rationale**:

- **10000 features**: Captures most important vocabulary without overfitting
- **Trigrams (1,3)**: Captures more context than bigrams (e.g., "if x ==", "for i in")
- **Word tokenization** (not character): Code tokens are meaningful units

> ⚠️ **Note on Character N-grams**: Character n-grams (3-5) can work for authorship attribution in prose, but explode in feature space for code. We use **word n-grams** instead.

**Implementation**:

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 3),      # Word unigrams, bigrams, and trigrams
    analyzer='word',          # Word-level, not character-level
    token_pattern=r'\b\w+\b', # Match word tokens
    min_df=2,                 # Ignore very rare terms
)

clf = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',  # Handle class imbalance
    solver='lbfgs',
)
```

**Expected Performance**: F1 ~0.60-0.70 (lexical baseline)

---

## 3. Preprocessing Strategy: The "Fingerprint Paradox"

### Decision: **Primary experiment uses RAW code** (no preprocessing)

> ⚠️ **Key Insight**: In MGC detection, the "logic" is often the most human-like part. The "style" (comments, whitespace, formatting) is where the AI's "signature" usually resides.

**The Paradox**:

- **Traditional NLP**: Strip noise (punctuation, casing) to focus on content
- **MGC Detection**: The "noise" IS the signal

**AI Fingerprints** (features we should KEEP):

- LLMs write "perfect" docstrings (consistent, standard-compliant)
- Uniform 4-space indentation (no mix of tabs/spaces)
- Comments like `"""This function computes..."""` vs human `# TODO: cleanup this mess`
- Standardized variable naming (snake_case everywhere)

**Human Fingerprints**:

- Inconsistent formatting, tabs mixed with spaces
- Informal comments: `# idk why this works`, `// HACK: fix later`
- Typos in comments and variable names
- Missing or sparse documentation

**Ablation Study Design**:

| Variant                | Description                           | Hypothesis                      |
| ---------------------- | ------------------------------------- | ------------------------------- |
| **Raw code** (primary) | No preprocessing                      | Preserves AI fingerprints       |
| Preprocessed           | Remove comments, normalize whitespace | Removes discriminative features |

**Prediction**: Raw code will outperform preprocessed code because we're preserving stylistic signals.

**Implementation** (preprocessing function kept for ablation):

```python
import re

def remove_comments(code: str, language: str = 'python') -> str:
    """Remove single-line and multi-line comments."""
    if language == 'python':
        code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)
        code = re.sub(r'""".*?"""', '', code, flags=re.DOTALL)
        code = re.sub(r"'''.*?'''", '', code, flags=re.DOTALL)
    elif language in ['java', 'c++', 'c']:
        code = re.sub(r'//.*$', '', code, flags=re.MULTILINE)
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    return code

def normalize_whitespace(code: str) -> str:
    """Normalize whitespace in code."""
    code = code.replace('\t', '    ')
    code = re.sub(r'[ \t]+$', '', code, flags=re.MULTILINE)
    code = re.sub(r'\n{3,}', '\n\n', code)
    return code.strip()
```

---

## 4. Training Configuration

### Decision: 3 Epochs, Batch Size 32, LR 2e-5

**Rationale**:

- **3 epochs**: Standard for fine-tuning transformers, prevents overfitting on small dataset
- **Batch 32**: Fits in GPU memory (16GB), provides stable gradients
- **LR 2e-5**: Standard for BERT-family fine-tuning
- **Warmup 10%**: Prevents early divergence

**Configuration**:

```python
training_args = {
    'batch_size': 32,
    'epochs': 3,
    'learning_rate': 2e-5,
    'warmup_ratio': 0.1,
    'max_length': 512,
    'weight_decay': 0.01,
}

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=2e-5,
    weight_decay=0.01,
)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps,
)
```

**GPU Memory Estimate**:

- CodeBERT: ~400MB
- Batch of 32 × 512 tokens: ~200MB
- Gradients/optimizer states: ~800MB
- Total: ~1.5GB (fits on most GPUs)

---

## 5. Evaluation Metrics

### Decision: F1 Score (Binary) as Primary, ROC-AUC for Analysis

**Rationale**:

- **F1 (primary)**: Kaggle leaderboard uses F1. Optimize for this.
- **ROC-AUC (analysis)**: Shows model discrimination capability across thresholds
- **Precision-Recall curve (poster)**: Richer analysis, good talking point

> **Note on False Positive Considerations**: In real-world MGC detection, false positives (accusing humans of using AI) are more damaging than false negatives. However, for this academic project, we optimize for F1 as that's what Kaggle evaluates.

**Implementation**:

```python
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    precision_recall_curve,
)

def evaluate_model(y_true, y_pred, y_proba=None):
    results = {
        'f1': f1_score(y_true, y_pred, average='binary'),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'confusion_matrix': confusion_matrix(y_true, y_pred),
    }
    if y_proba is not None:
        results['roc_auc'] = roc_auc_score(y_true, y_proba)
    return results
```

---

## 6. Data Handling: Multi-Label Stratification

### Decision: Stratify by BOTH label AND language

> ⚠️ **Critical Fix**: Stratifying only by label can lead to language imbalance (e.g., 80% Python in train, 80% Java in val). This produces unreliable metrics.

**Implementation**:

```python
from sklearn.model_selection import train_test_split

def create_stratified_subset(df, n_samples=20000, test_size=0.2):
    """Create stratified train/val split by label AND language."""
    # Sample n_samples with stratification
    if len(df) > n_samples:
        df = df.groupby(['label', 'language'], group_keys=False).apply(
            lambda x: x.sample(
                n=max(1, int(n_samples * len(x) / len(df))),
                random_state=42
            )
        )

    # Create combined stratification column
    df['stratify_col'] = df['label'].astype(str) + "_" + df['language']

    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df['stratify_col'],
        random_state=42,
    )

    # Drop helper column
    train_df = train_df.drop(columns=['stratify_col'])
    val_df = val_df.drop(columns=['stratify_col'])

    return train_df, val_df
```

**Verification**: After splitting, print language distribution in both train and val sets to confirm balance.

---

## 7. Visualization Standards

### Decision: Matplotlib + Seaborn with Consistent Style

**Rationale**:

- **Academic standard**: Clean, professional figures
- **Poster-ready**: High DPI, clear labels, appropriate font sizes
- **Consistent**: Same color scheme across all figures

**Style Configuration**:

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for all figures
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('colorblind')

# Figure settings
FIGURE_DPI = 300
FIGURE_SIZE = (8, 6)
FONT_SIZE = 12

def setup_figure():
    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=FIGURE_DPI)
    ax.tick_params(labelsize=FONT_SIZE)
    return fig, ax
```

**Color Scheme**:

- TF-IDF: Blue (#1f77b4)
- CodeBERT: Orange (#ff7f0e)
- Ablation (raw): Green (#2ca02c)
- Ablation (preprocessed): Gray (#7f7f7f)

---

## 8. Poster Template

### Decision: baposter LaTeX Class

**Rationale**:

- **A0 support**: Native A0 paper size
- **Column layout**: Easy multi-column academic poster format
- **Pragmatic**: Works, many examples available
- **Time constraint**: Learning a new tool (tikzposter, Canva) burns time we don't have

> **Philosophy**: A finished ugly poster beats an unfinished beautiful one.

**Template Structure**:

```latex
\documentclass[a0paper,portrait]{baposter}

\begin{document}
\begin{poster}{
    columns=3,
    headerheight=0.1\textheight,
    background=plain,
    bgColorOne=white,
}
{
    % Eye catcher (logo)
}
{
    % Title
    Detecting Machine-Generated Code with CodeBERT
}
{
    % Authors
    Anand Karna \& Zeev Tayer\\
    University of Bonn
}
{
    % Boxes...
}
\end{poster}
\end{document}
```

---

## Summary of Decisions

| Category          | Decision                         | Rationale                                  |
| ----------------- | -------------------------------- | ------------------------------------------ |
| Model             | CodeBERT-base (first 512 tokens) | Pre-trained on code, proven performance    |
| TF-IDF Config     | 10k features, word (1,3)-grams   | Standard baseline, captures token patterns |
| Preprocessing     | **Raw code (primary)**           | AI signatures live in comments/formatting  |
| Ablation          | Compare raw vs preprocessed      | Test the "fingerprint paradox" hypothesis  |
| Stratification    | **Label + Language**             | Prevents language bias in train/val        |
| Training          | 3 epochs, batch 32, lr 2e-5      | Standard fine-tuning                       |
| Primary Metric    | F1 (binary)                      | Matches Kaggle leaderboard                 |
| Secondary Metrics | ROC-AUC, PR curve                | Richer analysis for poster                 |
| Data Format       | Parquet, stratified              | Efficient, reproducible                    |
| Visualizations    | Matplotlib + Seaborn             | Academic standard, poster-ready            |
| Poster            | baposter class                   | Works, time-efficient                      |

---

## Key Hypotheses to Test

| Hypothesis              | Experiment            | Expected Result                 |
| ----------------------- | --------------------- | ------------------------------- |
| Raw code > preprocessed | Ablation study        | Raw code has higher F1          |
| CodeBERT > TF-IDF       | Baseline comparison   | 15-20% F1 improvement           |
| Python hardest language | Per-language analysis | Lower F1 (most human variation) |

---

## Resources

- **CodeBERT Paper**: [arXiv:2002.08155](https://arxiv.org/abs/2002.08155)
- **CodeBERT Model**: [microsoft/codebert-base](https://huggingface.co/microsoft/codebert-base)
- **Dataset**: [DaniilOr/SemEval-2026-Task13](https://huggingface.co/datasets/DaniilOr/SemEval-2026-Task13)
- **baposter**: [baposter documentation](http://www.brian-amberg.de/uni/poster/)
