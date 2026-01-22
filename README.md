# NLP Project Implementation Timeline (v3)

## SemEval-2026 Task 13: Detecting Machine-Generated Code

**Team 32:** Anand Karna & Zeev Tayer  
**Deadline:** January 24, 2026 (Poster Submission)

---

## Approach Summary

**What we're doing:**

- CodeBERT fine-tuning (strong baseline)
- TF-IDF + Logistic Regression (proves transformers are necessary)
- Rigorous analysis (per-language, per-generator, error analysis)
- One ablation study: raw code vs preprocessed (the "Fingerprint Paradox")

**What we're NOT doing:**

- ~~GraphCodeBERT~~ (risky, time-consuming, hypothesis is weak)
- ~~Complex architectures~~ (can't defend what we don't understand)

**Why this is academically stronger:**

- Complete story with multiple baselines
- Defensible claims with ablation evidence
- Deep analysis shows understanding, not just results

---

## Key Research Decisions

| Area                   | Decision                                         | Rationale                                   |
| ---------------------- | ------------------------------------------------ | ------------------------------------------- |
| **Stratification**     | By label AND language                            | Prevents language imbalance in train/val    |
| **Primary experiment** | Raw code (no preprocessing)                      | AI fingerprints live in comments/formatting |
| **TF-IDF config**      | 10k features, (1,3)-grams, min_df=2              | Trigrams capture more context               |
| **Training config**    | 3 epochs, lr=2e-5, warmup=10%, weight_decay=0.01 | Standard BERT fine-tuning                   |
| **Primary metric**     | F1 (binary)                                      | Matches Kaggle leaderboard                  |
| **Secondary metrics**  | ROC-AUC, PR curve                                | Richer analysis for poster                  |

### The "Fingerprint Paradox"

> ⚠️ **Key Insight**: In MGC detection, the "logic" is often the most human-like part. The "style" (comments, whitespace, formatting) is where the AI's "signature" usually resides.

**AI Fingerprints** (preserved in raw code):

- "Perfect" docstrings (consistent, standard-compliant)
- Uniform 4-space indentation
- Comments like `"""This function computes..."""`
- Standardized variable naming (snake_case everywhere)

**Human Fingerprints**:

- Inconsistent formatting, tabs mixed with spaces
- Informal comments: `# idk why this works`, `// HACK: fix later`
- Typos in comments and variable names

**Hypothesis**: Raw code will outperform preprocessed code.

---

## Overview

| Day   | Focus                         | End Goal                                       |
| ----- | ----------------------------- | ---------------------------------------------- |
| Day 1 | Data + Two Baselines          | TF-IDF F1 + CodeBERT F1 (both tasks)           |
| Day 2 | Ablation + Analysis + Visuals | All figures, error analysis, Kaggle submission |
| Day 3 | Poster                        | Final `poster.pdf` (A0)                        |

---

## DAY 1: Data + Two Baselines

| #   | Task                                                                 | Owner | Done When                                                | Time Est |
| --- | -------------------------------------------------------------------- | ----- | -------------------------------------------------------- | -------- |
| 1   | Download dataset, create 20k stratified subsets (A & B)              | Anand | `task_a_subset.parquet`, `task_b_subset.parquet` exist   | 30 min   |
|     | ⚠️ **Critical**: Stratify by BOTH label AND language                 |       | `stratify_col = label + "_" + language`                  |          |
| 2   | Verify stratification: print language distribution in train AND val  | Anand | Distributions match within 2%                            | 10 min   |
| 3   | Exploratory data analysis: class distribution, code length stats     | Zeev  | Printed stats: avg length, label counts, language counts | 30 min   |
| 4   | **Baseline 1:** TF-IDF + Logistic Regression (Task A)                | Anand | F1 score printed (expect ~0.60-0.70)                     | 1 hr     |
|     | Config: `max_features=10000, ngram_range=(1, 3), min_df=2`           |       |                                                          |          |
| 5   | **Baseline 1:** TF-IDF + Logistic Regression (Task B)                | Zeev  | Macro F1 score printed                                   | 1 hr     |
| 6   | Write preprocessing function (remove comments, normalize whitespace) | Anand | Function works on 100 samples                            | 1 hr     |
|     | Store BOTH `code` (raw) and `code_preprocessed` columns              |       |                                                          |          |
| 7   | Set up CodeBERT tokenizer + PyTorch DataLoader                       | Anand | Batch of 32 tokenizes to shape `(32, 512)`               | 1 hr     |
| 8   | **Baseline 2:** CodeBERT training script (Task A)                    | Anand | `train_task_a.py` runs 1 epoch without crash             | 2 hr     |
|     | Config: `lr=2e-5, weight_decay=0.01, warmup_ratio=0.1`               |       |                                                          |          |
| 9   | **Baseline 2:** CodeBERT training script (Task B)                    | Zeev  | `train_task_b.py` runs 1 epoch without crash             | 2 hr     |
| 10  | Train CodeBERT Task A on **RAW code** (3 epochs, 20k samples)        | Anand | Loss decreases, checkpoint saved                         | 1-2 hr   |
| 11  | Train CodeBERT Task B on **RAW code** (3 epochs, 20k samples)        | Zeev  | Loss decreases, checkpoint saved                         | 1-2 hr   |
| 12  | Evaluate both on validation set                                      | Both  | F1 + ROC-AUC for all models                              | 30 min   |

### ✅ Day 1 Checkpoint

> **You can fill this table:**
>
> | Model           | Task A F1 | Task A ROC-AUC | Task B Macro F1 |
> | --------------- | --------- | -------------- | --------------- |
> | TF-IDF + LogReg | \_\_\_    | \_\_\_         | \_\_\_          |
> | CodeBERT (raw)  | \_\_\_    | \_\_\_         | \_\_\_          |
> | **Improvement** | +\_\_\_%  |                | +\_\_\_%        |
>
> **Key insight:** "CodeBERT outperforms TF-IDF by \_\_\_%, proving semantic understanding matters."

---

## DAY 2: Ablation + Analysis + Visuals

| #   | Task                                                           | Owner  | Done When                                        | Time Est |
| --- | -------------------------------------------------------------- | ------ | ------------------------------------------------ | -------- |
| 13  | **Ablation:** Train CodeBERT WITH preprocessing                | Anand  | F1 with preprocessed code                        | 1.5 hr   |
|     | ⚠️ Use `--preprocess` flag (default is raw)                    |        |                                                  |          |
| 14  | Compare ablation results, confirm/reject "Fingerprint Paradox" | Anand  | Table filled, 1-sentence insight written         | 15 min   |
| 15  | Train final models on 30-50k with best config                  | Both   | `model_task_a_raw.pt`, `model_task_b.pt` saved   | 2 hr     |
| 16  | **Per-language F1** (Task A): Python, Java, C++ breakdown      | Anand  | Table with 3 rows                                | 30 min   |
| 17  | **Per-generator F1** (Task B): All 11 classes breakdown        | Zeev   | Table with 11 rows                               | 30 min   |
| 18  | Generate confusion matrix (Task A: 2×2)                        | Anand  | `confusion_matrix_task_a.png`                    | 20 min   |
| 19  | Generate confusion matrix (Task B: 11×11)                      | Zeev   | `confusion_matrix_task_b.png`                    | 20 min   |
| 20  | Generate training loss curves                                  | Either | `loss_curve_a.png`, `loss_curve_b.png`           | 20 min   |
| 21  | Generate PR curve (Task A)                                     | Anand  | `pr_curve_task_a.png`                            | 20 min   |
| 22  | **Error Analysis:** Manually inspect 30 misclassified samples  | Both   | Written notes: "Model fails when..."             | 1 hr     |
| 23  | Identify top 3 confusion pairs in Task B with explanation      | Zeev   | "Confuses X with Y because..."                   | 30 min   |
| 24  | Generate baseline comparison bar chart                         | Anand  | `baseline_comparison.png`                        | 20 min   |
| 25  | Generate Kaggle submission files                               | Both   | `submission_task_a.csv`, `submission_task_b.csv` | 30 min   |
| 26  | Submit to Kaggle                                               | Both   | Leaderboard scores recorded                      | 15 min   |

### ✅ Day 2 Checkpoint

> **All these artifacts exist:**
>
> - [ ] `confusion_matrix_task_a.png`
> - [ ] `confusion_matrix_task_b.png`
> - [ ] `loss_curve_a.png`
> - [ ] `loss_curve_b.png`
> - [ ] `baseline_comparison.png`
> - [ ] `pr_curve_task_a.png`
> - [ ] Ablation results table (raw vs preprocessed)
> - [ ] Per-language F1 table
> - [ ] Per-generator F1 table
> - [ ] Error analysis notes (30 samples)
> - [ ] Kaggle scores

---

## DAY 3: Poster

| #   | Task                                                      | Owner  | Done When                             | Time Est |
| --- | --------------------------------------------------------- | ------ | ------------------------------------- | -------- |
| 27  | Set up baposter A0 LaTeX template (3-column portrait)     | Anand  | `poster.tex` compiles to empty A0 PDF | 1 hr     |
| 28  | Write Introduction + Problem Statement (~150 words)       | Anand  | Text in poster                        | 45 min   |
| 29  | Write Methodology section + CodeBERT diagram              | Zeev   | Architecture diagram + 100 words      | 1 hr     |
| 30  | Insert baseline comparison figure + results tables        | Anand  | Shows TF-IDF vs CodeBERT              | 45 min   |
| 31  | Insert confusion matrices + per-language/generator tables | Zeev   | All figures render correctly          | 45 min   |
| 32  | Write Analysis section with 3 key findings                | Zeev   | Insights from error analysis          | 45 min   |
| 33  | Write Ablation Results ("Fingerprint Paradox" insight)    | Anand  | Table + explanation                   | 30 min   |
| 34  | Insert PR curve figure                                    | Anand  | Figure renders correctly              | 15 min   |
| 35  | Write Conclusion + Future Work (~100 words)               | Anand  | Text in poster                        | 30 min   |
| 36  | Add References (CodeBERT, Task13 papers)                  | Either | BibTeX renders correctly              | 15 min   |
| 37  | Final review: font sizes, layout, names visible           | Both   | Text readable from 1 meter            | 1 hr     |
| 38  | Export final `poster.pdf`                                 | Both   | A0 PDF, < 20MB, both names on it      | 15 min   |

### ✅ Day 3 Checkpoint

> **`poster.pdf` exists with these sections:**
>
> - [ ] Title + Names + Affiliation
> - [ ] Introduction / Problem Statement
> - [ ] Methodology (with diagram)
> - [ ] Baseline Comparison (TF-IDF vs CodeBERT)
> - [ ] Results (confusion matrices, F1 tables)
> - [ ] Ablation Study ("Fingerprint Paradox")
> - [ ] Analysis / Key Findings
> - [ ] Conclusion + Future Work
> - [ ] References

---

## Project Structure

```
project/
├── data/
│   ├── task_a_subset.parquet      # Contains both 'code' and 'code_preprocessed'
│   └── task_b_subset.parquet
├── src/
│   ├── preprocess.py               # Data loading + preprocessing
│   ├── baseline_tfidf.py           # TF-IDF + LogReg baseline
│   ├── train_task_a.py             # CodeBERT Task A (--preprocess flag)
│   ├── train_task_b.py             # CodeBERT Task B
│   ├── evaluate.py                 # Evaluation + metrics
│   └── error_analysis.py           # Manual inspection helper
├── outputs/
│   ├── models/
│   │   ├── model_task_a_raw.pt           # Primary: raw code
│   │   ├── model_task_a_preprocessed.pt  # Ablation: preprocessed
│   │   └── model_task_b.pt
│   ├── figures/
│   │   ├── confusion_matrix_task_a.png
│   │   ├── confusion_matrix_task_b.png
│   │   ├── loss_curve_a.png
│   │   ├── loss_curve_b.png
│   │   ├── baseline_comparison.png
│   │   └── pr_curve_task_a.png
│   └── submissions/
│       ├── submission_task_a.csv
│       └── submission_task_b.csv
├── poster/
│   ├── poster.tex
│   └── poster.pdf
└── notes/
    └── error_analysis.md           # Manual inspection notes
```

---

## Key Tables to Fill

### Baseline Comparison

| Model                        | Task A F1 | Task A ROC-AUC | Task B Macro F1 |
| ---------------------------- | --------- | -------------- | --------------- |
| TF-IDF + Logistic Regression | \_\_\_    | \_\_\_         | \_\_\_          |
| CodeBERT (raw code)          | \_\_\_    | \_\_\_         | \_\_\_          |
| **Improvement**              | +\_\_\_%  | +\_\_\_%       | +\_\_\_%        |

### Ablation Study: The "Fingerprint Paradox"

| Variant                 | Task A F1 | Notes                          |
| ----------------------- | --------- | ------------------------------ |
| **Raw code** (primary)  | \_\_\_    | Preserves AI fingerprints      |
| Preprocessed (ablation) | \_\_\_    | Removes comments/formatting    |
| **Difference**          | ±\_\_\_%  | Hypothesis: raw > preprocessed |

**Insight:** ************************\_************************

### Per-Language F1 (Task A)

| Language | Samples | F1     | Notes                                    |
| -------- | ------- | ------ | ---------------------------------------- |
| Python   | \_\_\_  | \_\_\_ | Expected: hardest (most human variation) |
| Java     | \_\_\_  | \_\_\_ |                                          |
| C++      | \_\_\_  | \_\_\_ |                                          |

### Per-Generator F1 (Task B)

| Generator   | Samples | F1     | Notes            |
| ----------- | ------- | ------ | ---------------- |
| Human       | \_\_\_  | \_\_\_ | (dominant class) |
| DeepSeek-AI | \_\_\_  | \_\_\_ |                  |
| Qwen        | \_\_\_  | \_\_\_ |                  |
| 01-ai       | \_\_\_  | \_\_\_ |                  |
| BigCode     | \_\_\_  | \_\_\_ |                  |
| Gemma       | \_\_\_  | \_\_\_ |                  |
| Phi         | \_\_\_  | \_\_\_ |                  |
| Meta-LLaMA  | \_\_\_  | \_\_\_ |                  |
| IBM-Granite | \_\_\_  | \_\_\_ |                  |
| Mistral     | \_\_\_  | \_\_\_ |                  |
| OpenAI      | \_\_\_  | \_\_\_ |                  |

### Error Analysis Summary

| Error Type | Count (/30) | Example | Hypothesis |
| ---------- | ----------- | ------- | ---------- |
| \_\_\_     | \_\_\_      | \_\_\_  | \_\_\_     |
| \_\_\_     | \_\_\_      | \_\_\_  | \_\_\_     |
| \_\_\_     | \_\_\_      | \_\_\_  | \_\_\_     |

---

## Visualization Standards

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Style configuration (use consistently)
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('colorblind')

FIGURE_DPI = 300
FIGURE_SIZE = (8, 6)
FONT_SIZE = 12

# Color scheme
COLORS = {
    'tfidf': '#1f77b4',       # Blue
    'codebert': '#ff7f0e',    # Orange
    'raw': '#2ca02c',         # Green
    'preprocessed': '#7f7f7f', # Gray
}
```

---

## Cross-Questioning Prep

### Questions You WILL Be Asked

| Question                                  | Your Answer                                                                                                                                                                          |
| ----------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| "Why CodeBERT over simpler models?"       | "TF-IDF achieved **_% F1, CodeBERT achieved _**% — the \_\_\_% improvement shows semantic understanding of code structure matters, not just keyword matching."                       |
| "Why raw code instead of preprocessing?"  | "The 'Fingerprint Paradox': AI signatures live in comments and formatting. Preprocessing removes discriminative features. Our ablation confirms raw code outperforms by \_\_\_% F1." |
| "What does preprocessing remove?"         | "Comment style (`"""docstrings"""` vs `# hack`), whitespace patterns (uniform vs inconsistent), naming conventions. These are AI fingerprints."                                      |
| "Which languages are hardest?"            | "Model struggles with **_ (F1=_**) because training data is Python-heavy and human Python code has the most stylistic variation."                                                    |
| "Which generators are hardest to detect?" | "**_ and _** are most confused (see confusion matrix) because they share similar base architectures."                                                                                |
| "What patterns does the model learn?"     | "From error analysis: model fails on [specific pattern]. It likely relies on [naming/structure/etc]."                                                                                |
| "What would you do differently?"          | "1) More balanced language distribution, 2) Longer context windows (sliding window pooling), 3) Ensemble with perplexity-based features."                                            |

### Questions About Things You DIDN'T Do

| Question                     | Your Answer                                                                                                                                                                                                     |
| ---------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| "Why not GraphCodeBERT?"     | "DFG extraction requires language-specific parsers and significant implementation time. Our hypothesis is that surface-level patterns (naming, comments) are more discriminative than data flow for this task." |
| "Why not larger dataset?"    | "We used 30k samples. CodeBERT saturates quickly due to pre-training — diminishing returns after ~50k samples for fine-tuning."                                                                                 |
| "Why not ensemble?"          | "Time constraint. Future work could combine CodeBERT with perplexity-based features from generative models like StarCoder2."                                                                                    |
| "Why first 512 tokens only?" | "CodeBERT's context limit. AI patterns typically appear in function headers/docstrings at the start. Sliding window pooling is future work."                                                                    |

---

## Code Templates

### TF-IDF Baseline (baseline_tfidf.py)

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
import pandas as pd

def train_tfidf_baseline(train_df, val_df, task="A"):
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 3),      # Trigrams for more context
        analyzer='word',
        token_pattern=r'\b\w+\b',
        min_df=2,                # Ignore very rare terms
    )
    X_train = vectorizer.fit_transform(train_df['code'])  # Use raw code
    X_val = vectorizer.transform(val_df['code'])

    clf = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        solver='lbfgs',
    )
    clf.fit(X_train, train_df['label'])

    preds = clf.predict(X_val)
    proba = clf.predict_proba(X_val)[:, 1] if task == "A" else None

    if task == "A":
        f1 = f1_score(val_df['label'], preds, average='binary')
        roc_auc = roc_auc_score(val_df['label'], proba)
        return f1, roc_auc, clf, vectorizer
    else:
        f1 = f1_score(val_df['label'], preds, average='macro')
        return f1, clf, vectorizer
```

### Stratified Data Split (preprocess.py)

```python
from sklearn.model_selection import train_test_split

def create_stratified_subset(df, n_samples=20000, test_size=0.2):
    """Create stratified train/val split by BOTH label AND language."""
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

    # Verify distribution
    print("Train language dist:", train_df['language'].value_counts(normalize=True))
    print("Val language dist:", val_df['language'].value_counts(normalize=True))

    return train_df, val_df
```

### CodeBERT Classifier (train_task_a.py)

```python
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

class CodeBERTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.codebert = AutoModel.from_pretrained("microsoft/codebert-base")
        self.classifier = nn.Linear(768, 2)  # binary classification

    def forward(self, input_ids, attention_mask):
        outputs = self.codebert(input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0]  # [CLS] token
        return self.classifier(cls_output)

# Training config
training_config = {
    'batch_size': 32,
    'epochs': 3,
    'learning_rate': 2e-5,
    'warmup_ratio': 0.1,
    'max_length': 512,
    'weight_decay': 0.01,
}

# Optimizer and scheduler setup
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=training_config['learning_rate'],
    weight_decay=training_config['weight_decay'],
)

total_steps = len(train_loader) * training_config['epochs']
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(training_config['warmup_ratio'] * total_steps),
    num_training_steps=total_steps,
)
```

### Error Analysis Helper (error_analysis.py)

```python
import pandas as pd

def get_misclassified_samples(val_df, predictions, n=30):
    """Get n misclassified samples for manual inspection."""
    val_df = val_df.copy()
    val_df['pred'] = predictions
    val_df['correct'] = val_df['label'] == val_df['pred']

    misclassified = val_df[~val_df['correct']].sample(n=min(n, len(val_df[~val_df['correct']])))

    for idx, row in misclassified.iterrows():
        print("=" * 80)
        print(f"TRUE: {row.get('generator', 'N/A')} (label={row['label']})")
        print(f"PRED: label={row['pred']}")
        print(f"LANGUAGE: {row['language']}")
        print("-" * 40)
        print(row['code'][:500])  # First 500 chars
        print("=" * 80)
        input("Press Enter for next sample...")
```

---

## Communication Protocol

| Time         | Sync Action                                                       |
| ------------ | ----------------------------------------------------------------- |
| End of Day 1 | Share: All F1 + ROC-AUC numbers (TF-IDF A/B, CodeBERT A/B)        |
| Mid Day 2    | Share: Ablation results, confirm "Fingerprint Paradox" hypothesis |
| End of Day 2 | Share: All PNGs (including PR curve), error analysis notes        |
| Mid Day 3    | Review poster draft together, practice Q&A                        |

---

## Key Resources

- **Dataset:** [HuggingFace](https://huggingface.co/datasets/DaniilOr/SemEval-2026-Task13)
- **GitHub:** [mbzuai-nlp/SemEval-2026-Task13](https://github.com/mbzuai-nlp/SemEval-2026-Task13)
- **Kaggle Task A:** [Competition Link](https://www.kaggle.com/t/99673e23fe8546cf9a07a40f36f2cc7e)
- **Kaggle Task B:** [Competition Link](https://www.kaggle.com/t/65af9e22be6d43d884cfd6e41cad3ee4)
- **CodeBERT:** [microsoft/codebert-base](https://huggingface.co/microsoft/codebert-base)
- **CodeBERT Paper:** [arXiv:2002.08155](https://arxiv.org/abs/2002.08155)
- **baposter:** [baposter documentation](http://www.brian-amberg.de/uni/poster/)

---

## Hypotheses to Validate

| Hypothesis              | Experiment            | Expected Result            |
| ----------------------- | --------------------- | -------------------------- |
| Raw code > preprocessed | Ablation study        | Raw code has higher F1     |
| CodeBERT > TF-IDF       | Baseline comparison   | 15-20% F1 improvement      |
| Python hardest language | Per-language analysis | Lowest F1 (most variation) |

---

## Academic Narrative (For Poster/Presentation)

> **Story arc:**
>
> 1. **Problem:** LLMs generate increasingly human-like code, threatening academic integrity and software supply chain security.
> 2. **Challenge:** Detection must generalize across languages (Python, Java, C++, Go, PHP) and generator families (10 LLM types).
> 3. **Approach:** We compare TF-IDF (lexical baseline) with CodeBERT (semantic understanding) to quantify the importance of code semantics.
> 4. **Key Insight — The "Fingerprint Paradox":** Traditional NLP preprocessing removes noise to focus on content. But in MGC detection, the "noise" IS the signal. AI-generated code has distinctive patterns in comments, formatting, and whitespace that human code lacks.
> 5. **Findings:**
>    - CodeBERT outperforms TF-IDF by \_\_\_%, confirming semantic features matter
>    - Raw code outperforms preprocessed by \_\_\_%, validating the "Fingerprint Paradox"
>    - Model struggles with [language] and confuses [generator pairs]
> 6. **Conclusion:** Pre-trained code models are effective, and preserving stylistic features is crucial for detection. Future work should explore cross-lingual transfer and ensemble with perplexity-based features.

This is a **complete, defensible academic story** — not just "we got good numbers."
