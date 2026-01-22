# Implementation Plan: Machine-Generated Code Detection (Anand)

**Branch**: `002-anand-mgc-detection` | **Date**: 2026-01-22 | **Spec**: [spec.md](./spec.md)  
**Deadline**: January 24, 2026 (Poster Submission)

---

## Summary

Implement binary classification (Task A) for detecting machine-generated code using two approaches: TF-IDF baseline and CodeBERT fine-tuning. The project includes ablation study (raw vs preprocessed code), per-language analysis, and A0 poster creation. All work targets demonstrating that semantic understanding (CodeBERT) outperforms lexical features (TF-IDF).

---

## Technical Context

**Language/Version**: Python 3.9+  
**Primary Dependencies**:

- PyTorch 2.x (with CUDA for GPU training)
- transformers (Hugging Face)
- scikit-learn (TF-IDF, Logistic Regression, metrics)
- pandas (data handling)
- matplotlib/seaborn (visualizations)
- pyarrow (parquet support)

**Hardware**: GPU recommended (Google Colab Pro or university cluster)  
**Target**: Kaggle submission + A0 poster

**Performance Goals**:

- TF-IDF training: <5 minutes
- CodeBERT training: 1-2 hours (3 epochs × 20k samples)
- Inference: <1 second per sample

---

## Project Structure

```text
project/
├── data/
│   └── task_a_subset.parquet          # 20k stratified samples
├── src/
│   ├── preprocess.py                  # Data loading + preprocessing
│   ├── baseline_tfidf.py              # TF-IDF + LogReg baseline
│   ├── train_task_a.py                # CodeBERT training
│   ├── evaluate.py                    # Evaluation + metrics
│   └── error_analysis.py              # Manual inspection helper
├── outputs/
│   ├── models/
│   │   ├── model_task_a_raw.pt        # Primary: trained on raw code
│   │   └── model_task_a_preprocessed.pt  # Ablation: trained on preprocessed
│   ├── figures/
│   │   ├── confusion_matrix_task_a.png
│   │   ├── loss_curve_a.png
│   │   ├── baseline_comparison.png
│   │   └── pr_curve_task_a.png
│   └── submissions/
│       └── submission_task_a.csv
├── poster/
│   ├── poster.tex
│   └── poster.pdf
└── notes/
    └── error_analysis.md
```

---

## Phase 0: Environment Setup

### Setup Tasks

1. **Create virtual environment and install dependencies**

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Verify GPU availability** (for CodeBERT training)

   ```python
   import torch
   print(torch.cuda.is_available())  # Should be True
   ```

3. **Verify HuggingFace dataset access**
   ```python
   from datasets import load_dataset
   ds = load_dataset("DaniilOr/SemEval-2026-Task13", split="train[:10]")
   print(ds)  # Should print dataset info
   ```

---

## Phase 1: Data Preparation (Day 1, Tasks 1 & 5)

### Implementation

#### [NEW] src/preprocess.py

**Purpose**: Download dataset, create stratified subset, preprocessing functions

```python
# Key functions to implement:
def download_dataset() -> Dataset:
    """Download SemEval-2026 Task 13 dataset from HuggingFace."""

def create_stratified_subset(df, n_samples=20000, test_size=0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create stratified train/val split by BOTH label AND language.

    Critical: Stratifying only by label can lead to language imbalance.
    """
    # Create combined stratification column
    df['stratify_col'] = df['label'].astype(str) + "_" + df['language']

    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df['stratify_col'],
        random_state=42,
    )
    return train_df.drop(columns=['stratify_col']), val_df.drop(columns=['stratify_col'])

def remove_comments(code: str, language: str) -> str:
    """Remove single-line and multi-line comments based on language."""

def normalize_whitespace(code: str) -> str:
    """Normalize tabs, multiple spaces, trailing whitespace."""

def preprocess_code(code: str, language: str = 'python') -> str:
    """Full preprocessing pipeline: comments + whitespace."""
```

**Output**: `data/task_a_subset.parquet` with columns:

- `code`: Raw code string (PRIMARY - preserves AI fingerprints)
- `code_preprocessed`: Preprocessed code (for ablation only)
- `label`: 0 (human) or 1 (AI-generated)
- `language`: Python, Java, C++, etc.

> ⚠️ **Key Insight**: In MGC detection, the "noise" IS the signal. AI fingerprints live in comments, formatting, and whitespace patterns. Raw code is the primary experiment.

---

## Phase 2: TF-IDF Baseline (Day 1, Task 3)

### Implementation

#### [NEW/MODIFY] src/baseline_tfidf.py

**Purpose**: Implement TF-IDF + Logistic Regression baseline

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def train_tfidf_baseline(train_df, val_df) -> Tuple[float, LogisticRegression, TfidfVectorizer]:
    """
    Train TF-IDF + Logistic Regression for Task A (binary classification).

    Config (from research):
    - TfidfVectorizer: max_features=10000, ngram_range=(1, 3), min_df=2
    - LogisticRegression: max_iter=1000, class_weight='balanced', solver='lbfgs'

    Returns: (f1_score, trained_model, fitted_vectorizer)
    """
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 3),      # Word unigrams, bigrams, AND trigrams
        analyzer='word',
        token_pattern=r'\b\w+\b',
        min_df=2,                # Ignore very rare terms
    )

    clf = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        solver='lbfgs',
    )
    # ... training code
```

**Expected Output**: F1 score ~0.60-0.70 (lexical baseline)

---

## Phase 3: CodeBERT Implementation (Day 1, Tasks 6, 7, 9)

### Implementation

#### [NEW] src/train_task_a.py

**Purpose**: CodeBERT fine-tuning for Task A

```python
# Key components:
class CodeDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for tokenized code samples."""

def create_dataloader(df, tokenizer, batch_size=32, max_length=512):
    """Create DataLoader with tokenized samples."""

class CodeBERTClassifier(nn.Module):
    """CodeBERT with binary classification head."""
    def __init__(self):
        super().__init__()
        self.codebert = AutoModel.from_pretrained("microsoft/codebert-base")
        self.classifier = nn.Linear(768, 2)  # binary classification

    def forward(self, input_ids, attention_mask):
        outputs = self.codebert(input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0]  # [CLS] token
        return self.classifier(cls_output)

def train_epoch(model, dataloader, optimizer, scheduler, device):
    """Train one epoch, return average loss."""

def evaluate(model, dataloader, device):
    """Evaluate model, return metrics dict and predictions."""

def main():
    """Main training loop: load data, train 3 epochs, save checkpoint."""
```

**Training Config** (from research):

| Parameter     | Value                     | Rationale                          |
| ------------- | ------------------------- | ---------------------------------- |
| Model         | `microsoft/codebert-base` | Pre-trained on code                |
| Max length    | 512 tokens                | CodeBERT limit; first 512 strategy |
| Batch size    | 32                        | Fits 16GB GPU, stable gradients    |
| Epochs        | 3                         | Prevents overfitting               |
| Learning rate | 2e-5                      | Standard for BERT fine-tuning      |
| Warmup ratio  | 0.1                       | Prevents early divergence          |
| Weight decay  | 0.01                      | Regularization                     |
| Optimizer     | AdamW                     | Standard for transformers          |
| Scheduler     | Linear with warmup        | Gradual LR decay                   |

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps,
)
```

**Output**:

- `outputs/models/model_task_a_raw.pt` (primary)
- Training loss per epoch (for loss curve)

---

## Phase 4: Ablation Study (Day 2, Task 12)

### The "Fingerprint Paradox"

> ⚠️ **Key Insight**: In MGC detection, the "logic" is often the most human-like part. The "style" (comments, whitespace, formatting) is where the AI's "signature" usually resides.

**AI Fingerprints** (features preserved in raw code):

- "Perfect" docstrings (consistent, standard-compliant)
- Uniform 4-space indentation
- Comments like `"""This function computes..."""`
- Standardized variable naming

**Human Fingerprints**:

- Inconsistent formatting, tabs mixed with spaces
- Informal comments: `# idk why this works`, `// HACK: fix later`
- Typos in comments and variable names

### Implementation

Run CodeBERT training **twice**:

1. **Primary**: Raw code (no preprocessing) — expected to perform BETTER
2. **Ablation**: Preprocessed code (comment removal + whitespace normalization)

**Modification to train_task_a.py**:

```python
# Default is now RAW code (--preprocess flag enables preprocessing)
parser.add_argument('--preprocess', action='store_true',
                    help='Use preprocessed code (ablation study)')
```

**Run commands**:

```bash
# Primary experiment: RAW code (default)
python src/train_task_a.py --output outputs/models/model_task_a_raw.pt

# Ablation: WITH preprocessing
python src/train_task_a.py --preprocess --output outputs/models/model_task_a_preprocessed.pt
```

**Hypothesis**: Raw code will outperform preprocessed code because we're preserving stylistic signals.

**Output**: Ablation comparison table

| Variant      | F1 Score | Notes                           |
| ------------ | -------- | ------------------------------- |
| Raw code     | TBD      | Preserves AI fingerprints       |
| Preprocessed | TBD      | Removes discriminative features |

---

## Phase 5: Analysis & Visualizations (Day 2, Tasks 14, 16, 21)

### Implementation

#### [NEW] src/evaluate.py

**Purpose**: Comprehensive evaluation and visualization generation

```python
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
    roc_auc_score, precision_recall_curve,
)

def evaluate_model(y_true, y_pred, y_proba=None) -> dict:
    """Compute all metrics."""
    results = {
        'f1': f1_score(y_true, y_pred, average='binary'),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'confusion_matrix': confusion_matrix(y_true, y_pred),
    }
    if y_proba is not None:
        results['roc_auc'] = roc_auc_score(y_true, y_proba)
    return results

def compute_per_language_f1(df, predictions) -> pd.DataFrame:
    """Compute F1 for each language (Python, Java, C++)."""

def generate_confusion_matrix(y_true, y_pred, save_path):
    """Generate and save 2×2 confusion matrix visualization."""

def generate_loss_curve(train_losses, save_path):
    """Generate and save training loss curve."""

def generate_baseline_comparison(results_dict, save_path):
    """Generate bar chart comparing TF-IDF vs CodeBERT."""

def generate_pr_curve(y_true, y_proba, save_path):
    """Generate precision-recall curve for poster."""
```

**Visualization Standards** (from research):

```python
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('colorblind')

FIGURE_DPI = 300
FIGURE_SIZE = (8, 6)
FONT_SIZE = 12

# Color scheme
COLORS = {
    'tfidf': '#1f77b4',      # Blue
    'codebert': '#ff7f0e',   # Orange
    'raw': '#2ca02c',        # Green
    'preprocessed': '#7f7f7f', # Gray
}
```

**Outputs**:

- `outputs/figures/confusion_matrix_task_a.png`
- `outputs/figures/loss_curve_a.png`
- `outputs/figures/baseline_comparison.png`
- `outputs/figures/pr_curve_task_a.png`
- Per-language F1 table

**Metrics Summary**:

| Metric   | Role      | Notes                        |
| -------- | --------- | ---------------------------- |
| F1       | Primary   | Kaggle leaderboard metric    |
| ROC-AUC  | Secondary | Model discrimination ability |
| PR Curve | Poster    | Richer analysis, good visual |

---

## Phase 6: Poster Creation (Day 3, Tasks 24, 25, 27, 30, 31)

### Implementation

#### [NEW] poster/poster.tex

**Purpose**: A0 academic poster using baposter class

**Poster Sections (Anand's responsibility)**:

1. **Title + Names** - Team 32: Anand Karna & Zeev Tayer
2. **Introduction** (~150 words) - Problem statement, motivation
3. **Baseline Comparison Figure** - TF-IDF vs CodeBERT bar chart
4. **Results Tables** - Main results + per-language breakdown
5. **Ablation Results** - Preprocessing impact + "fingerprint paradox" insight
6. **Conclusion** (~100 words) - Summary + future work

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
% ... poster content
\end{poster}
\end{document}
```

> **Philosophy**: A finished ugly poster beats an unfinished beautiful one.

---

## Verification Plan

### Automated Tests

1. **Data Preparation Test**

   ```bash
   python -c "
   import pandas as pd
   df = pd.read_parquet('data/task_a_subset.parquet')
   assert len(df) == 20000
   assert 'code' in df.columns
   assert 'code_preprocessed' in df.columns
   print('✓ Data subset verified')
   "
   ```

2. **Stratification Verification** (critical)

   ```bash
   python -c "
   import pandas as pd
   train = pd.read_parquet('data/train.parquet')
   val = pd.read_parquet('data/val.parquet')
   # Check language distribution is similar
   print('Train language dist:', train['language'].value_counts(normalize=True))
   print('Val language dist:', val['language'].value_counts(normalize=True))
   print('✓ Stratification verified')
   "
   ```

3. **Preprocessing Test**

   ```bash
   python -c "
   from src.preprocess import remove_comments, normalize_whitespace
   code = '# comment\nprint(1)  '
   result = remove_comments(code, 'python')
   assert '# comment' not in result
   print('✓ Preprocessing verified')
   "
   ```

4. **TF-IDF Baseline Test**

   ```bash
   python src/baseline_tfidf.py --test  # Should print F1 score
   ```

5. **CodeBERT Tokenization Test**

   ```bash
   python -c "
   from transformers import AutoTokenizer
   tokenizer = AutoTokenizer.from_pretrained('microsoft/codebert-base')
   batch = ['print(1)', 'return x']
   tokens = tokenizer(batch, padding='max_length', max_length=512, truncation=True, return_tensors='pt')
   assert tokens['input_ids'].shape == (2, 512)
   print('✓ Tokenization verified')
   "
   ```

6. **Training Smoke Test** (1 epoch)
   ```bash
   python src/train_task_a.py --epochs 1 --samples 1000 --test-mode
   # Should complete without crash, print loss
   ```

### Manual Verification

1. **Visual Check**: Open generated PNG files and verify they look correct
2. **Poster Compile**: Run `pdflatex poster.tex` and check PDF opens correctly
3. **Kaggle Submission**: Upload `submission_task_a.csv` and verify it's accepted

---

## Key Hypotheses to Test

| Hypothesis              | Experiment            | Expected Result                 |
| ----------------------- | --------------------- | ------------------------------- |
| Raw code > preprocessed | Ablation study        | Raw code has higher F1          |
| CodeBERT > TF-IDF       | Baseline comparison   | 15-20% F1 improvement           |
| Python hardest language | Per-language analysis | Lower F1 (most human variation) |

---

## Timeline

| Phase   | Tasks                      | Duration | Done When                                              |
| ------- | -------------------------- | -------- | ------------------------------------------------------ |
| Phase 1 | Data prep + stratification | 1.5 hr   | `task_a_subset.parquet` exists, language dist verified |
| Phase 2 | TF-IDF baseline            | 1 hr     | F1 score ~0.60-0.70 printed                            |
| Phase 3 | CodeBERT training (raw)    | 3-4 hr   | Checkpoint saved, loss decreases                       |
| Phase 4 | Ablation (preprocessed)    | 2 hr     | Two F1 scores compared, raw wins                       |
| Phase 5 | Analysis + figures         | 2 hr     | All PNGs generated, metrics logged                     |
| Phase 6 | Poster                     | 4-5 hr   | `poster.pdf` exists                                    |

**Total**: ~14-16 hours across 3 days

---

## Resources

- **CodeBERT Paper**: [arXiv:2002.08155](https://arxiv.org/abs/2002.08155)
- **CodeBERT Model**: [microsoft/codebert-base](https://huggingface.co/microsoft/codebert-base)
- **Dataset**: [DaniilOr/SemEval-2026-Task13](https://huggingface.co/datasets/DaniilOr/SemEval-2026-Task13)
- **baposter**: [baposter documentation](http://www.brian-amberg.de/uni/poster/)

---

## Next Steps

1. ✅ Review this plan
2. Execute Phase 1: Data preparation with proper stratification
3. Execute Phase 2: TF-IDF baseline (trigrams config)
4. Execute Phase 3: CodeBERT on RAW code (primary)
5. Execute Phase 4: Ablation on preprocessed code
6. Execute Phase 5: Analysis + all figures
7. Execute Phase 6: Poster creation
8. Submit to Kaggle + export final poster
