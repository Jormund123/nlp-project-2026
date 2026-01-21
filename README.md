# NLP Project Implementation Timeline (v2)
## SemEval-2026 Task 13: Detecting Machine-Generated Code
**Team 32:** Anand Karna & Zeev Tayer  
**Deadline:** January 24, 2026 (Poster Submission)

---

## Approach Summary

**What we're doing:**
- CodeBERT fine-tuning (strong baseline)
- TF-IDF + Logistic Regression (proves transformers are necessary)
- Rigorous analysis (per-language, per-generator, error analysis)
- One ablation study (with/without comment removal)

**What we're NOT doing:**
- ~~GraphCodeBERT~~ (risky, time-consuming, hypothesis is weak)
- ~~Complex architectures~~ (can't defend what we don't understand)

**Why this is academically stronger:**
- Complete story with multiple baselines
- Defensible claims with ablation evidence
- Deep analysis shows understanding, not just results

---

## Overview

| Day | Focus | End Goal |
|-----|-------|----------|
| Day 1 | Data + Two Baselines | TF-IDF F1 + CodeBERT F1 (both tasks) |
| Day 2 | Ablation + Analysis + Visuals | All figures, error analysis, Kaggle submission |
| Day 3 | Poster | Final `poster.pdf` (A0) |

---

## DAY 1: Data + Two Baselines

| # | Task | Owner | Done When | Time Est |
|---|------|-------|-----------|----------|
| 1 | Download dataset, create 20k stratified subsets (A & B) | Anand | `task_a_subset.parquet`, `task_b_subset.parquet` exist | 30 min |
| 2 | Exploratory data analysis: class distribution, code length stats | Zeev | Printed stats: avg length, label counts, language counts | 30 min |
| 3 | **Baseline 1:** TF-IDF + Logistic Regression (Task A) | Anand | F1 score printed (expect ~0.60-0.70) | 1 hr |
| 4 | **Baseline 1:** TF-IDF + Logistic Regression (Task B) | Zeev | Macro F1 score printed | 1 hr |
| 5 | Write preprocessing function (remove comments, normalize whitespace) | Anand | Function works on 100 samples | 1 hr |
| 6 | Set up CodeBERT tokenizer + PyTorch DataLoader | Anand | Batch of 32 tokenizes to shape `(32, 512)` | 1 hr |
| 7 | **Baseline 2:** CodeBERT training script (Task A) | Anand | `train_task_a.py` runs 1 epoch without crash | 2 hr |
| 8 | **Baseline 2:** CodeBERT training script (Task B) | Zeev | `train_task_b.py` runs 1 epoch without crash | 2 hr |
| 9 | Train CodeBERT Task A (3 epochs, 20k samples) | Anand | Loss decreases, checkpoint saved | 1-2 hr |
| 10 | Train CodeBERT Task B (3 epochs, 20k samples) | Zeev | Loss decreases, checkpoint saved | 1-2 hr |
| 11 | Evaluate both on validation set | Both | Four F1 numbers: TF-IDF A, TF-IDF B, CodeBERT A, CodeBERT B | 30 min |

### ✅ Day 1 Checkpoint
> **You can fill this table:**
>
> | Model | Task A F1 | Task B Macro F1 |
> |-------|-----------|-----------------|
> | TF-IDF + LogReg | ___ | ___ |
> | CodeBERT | ___ | ___ |
>
> **Key insight:** "CodeBERT outperforms TF-IDF by ___%, proving semantic understanding matters."

---

## DAY 2: Ablation + Analysis + Visuals

| # | Task | Owner | Done When | Time Est |
|---|------|-------|-----------|----------|
| 12 | **Ablation:** Train CodeBERT WITHOUT preprocessing (raw code) | Anand | F1 with raw code vs preprocessed code | 1.5 hr |
| 13 | Train final models on 30-50k with best config | Both | `model_task_a.pt`, `model_task_b.pt` saved | 2 hr |
| 14 | **Per-language F1** (Task A): Python, Java, C++ breakdown | Anand | Table with 3 rows | 30 min |
| 15 | **Per-generator F1** (Task B): All 11 classes breakdown | Zeev | Table with 11 rows | 30 min |
| 16 | Generate confusion matrix (Task A: 2×2) | Anand | `confusion_matrix_task_a.png` | 20 min |
| 17 | Generate confusion matrix (Task B: 11×11) | Zeev | `confusion_matrix_task_b.png` | 20 min |
| 18 | Generate training loss curves | Either | `loss_curve_a.png`, `loss_curve_b.png` | 20 min |
| 19 | **Error Analysis:** Manually inspect 30 misclassified samples | Both | Written notes: "Model fails when..." | 1 hr |
| 20 | Identify top 3 confusion pairs in Task B with explanation | Zeev | "Confuses X with Y because..." | 30 min |
| 21 | Generate baseline comparison bar chart | Anand | `baseline_comparison.png` | 20 min |
| 22 | Generate Kaggle submission files | Both | `submission_task_a.csv`, `submission_task_b.csv` | 30 min |
| 23 | Submit to Kaggle | Both | Leaderboard scores recorded | 15 min |

### ✅ Day 2 Checkpoint
> **All these artifacts exist:**
> - [ ] `confusion_matrix_task_a.png`
> - [ ] `confusion_matrix_task_b.png`
> - [ ] `loss_curve_a.png`
> - [ ] `loss_curve_b.png`
> - [ ] `baseline_comparison.png`
> - [ ] Ablation results table
> - [ ] Per-language F1 table
> - [ ] Per-generator F1 table
> - [ ] Error analysis notes (30 samples)
> - [ ] Kaggle scores

---

## DAY 3: Poster

| # | Task | Owner | Done When | Time Est |
|---|------|-------|-----------|----------|
| 24 | Set up A0 LaTeX poster template | Anand | `poster.tex` compiles to empty A0 PDF | 1 hr |
| 25 | Write Introduction + Problem Statement (~150 words) | Anand | Text in poster | 45 min |
| 26 | Write Methodology section + CodeBERT diagram | Zeev | Architecture diagram + 100 words | 1 hr |
| 27 | Insert baseline comparison figure + results tables | Anand | Shows TF-IDF vs CodeBERT | 45 min |
| 28 | Insert confusion matrices + per-language/generator tables | Zeev | All figures render correctly | 45 min |
| 29 | Write Analysis section with 3 key findings | Zeev | Insights from error analysis | 45 min |
| 30 | Write Ablation Results (preprocessing impact) | Anand | Table + 1 sentence insight | 30 min |
| 31 | Write Conclusion + Future Work (~100 words) | Anand | Text in poster | 30 min |
| 32 | Add References (CodeBERT, Task13 papers) | Either | BibTeX renders correctly | 15 min |
| 33 | Final review: font sizes, layout, names visible | Both | Text readable from 1 meter | 1 hr |
| 34 | Export final `poster.pdf` | Both | A0 PDF, < 20MB, both names on it | 15 min |

### ✅ Day 3 Checkpoint
> **`poster.pdf` exists with these sections:**
> - [ ] Title + Names + Affiliation
> - [ ] Introduction / Problem Statement
> - [ ] Methodology (with diagram)
> - [ ] Baseline Comparison (TF-IDF vs CodeBERT)
> - [ ] Results (confusion matrices, F1 tables)
> - [ ] Ablation Study (preprocessing impact)
> - [ ] Analysis / Key Findings
> - [ ] Conclusion + Future Work
> - [ ] References

---

## Project Structure

```
project/
├── data/
│   ├── task_a_subset.parquet
│   └── task_b_subset.parquet
├── src/
│   ├── preprocess.py               # Data loading + preprocessing
│   ├── baseline_tfidf.py           # TF-IDF + LogReg baseline
│   ├── train_task_a.py             # CodeBERT Task A
│   ├── train_task_b.py             # CodeBERT Task B
│   ├── evaluate.py                 # Evaluation + metrics
│   └── error_analysis.py           # Manual inspection helper
├── outputs/
│   ├── models/
│   │   ├── model_task_a.pt
│   │   └── model_task_b.pt
│   ├── figures/
│   │   ├── confusion_matrix_task_a.png
│   │   ├── confusion_matrix_task_b.png
│   │   ├── loss_curve_a.png
│   │   ├── loss_curve_b.png
│   │   └── baseline_comparison.png
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
| Model | Task A F1 | Task B Macro F1 |
|-------|-----------|-----------------|
| TF-IDF + Logistic Regression | ___ | ___ |
| CodeBERT (fine-tuned) | ___ | ___ |
| **Improvement** | +___% | +___% |

### Ablation Study (Preprocessing Impact)
| Preprocessing | Task A F1 | Task B Macro F1 |
|---------------|-----------|-----------------|
| Raw code (no preprocessing) | ___ | ___ |
| With comment removal + whitespace normalization | ___ | ___ |
| **Difference** | ±___% | ±___% |

### Per-Language F1 (Task A)
| Language | Samples | F1 | Notes |
|----------|---------|-----|-------|
| Python | ___ | ___ | (most data) |
| Java | ___ | ___ | |
| C++ | ___ | ___ | |

### Per-Generator F1 (Task B)
| Generator | Samples | F1 | Notes |
|-----------|---------|-----|-------|
| Human | ___ | ___ | (dominant class) |
| DeepSeek-AI | ___ | ___ | |
| Qwen | ___ | ___ | |
| 01-ai | ___ | ___ | |
| BigCode | ___ | ___ | |
| Gemma | ___ | ___ | |
| Phi | ___ | ___ | |
| Meta-LLaMA | ___ | ___ | |
| IBM-Granite | ___ | ___ | |
| Mistral | ___ | ___ | |
| OpenAI | ___ | ___ | |

### Error Analysis Summary
| Error Type | Count (/30) | Example | Hypothesis |
|------------|-------------|---------|------------|
| ___ | ___ | ___ | ___ |
| ___ | ___ | ___ | ___ |
| ___ | ___ | ___ | ___ |

---

## Cross-Questioning Prep

### Questions You WILL Be Asked

| Question | Your Answer |
|----------|-------------|
| "Why CodeBERT over simpler models?" | "TF-IDF achieved ___% F1, CodeBERT achieved ___% — the ___% improvement shows semantic understanding of code structure matters, not just keyword matching." |
| "What does preprocessing contribute?" | "Our ablation shows preprocessing improves/hurts F1 by ___%. Comment removal helps because..." |
| "Which languages are hardest?" | "Model struggles with ___ (F1=___) because training data is Python-heavy." |
| "Which generators are hardest to detect?" | "___ and ___ are most confused (see confusion matrix) because they share similar base architectures." |
| "What patterns does the model learn?" | "From error analysis: model fails on [specific pattern]. It likely relies on [naming/structure/etc]." |
| "What would you do differently?" | "1) More balanced language distribution, 2) Longer context windows, 3) Ensemble with stylometric features." |

### Questions About Things You DIDN'T Do

| Question | Your Answer |
|----------|-------------|
| "Why not GraphCodeBERT?" | "DFG extraction requires language-specific parsers and significant implementation time. Our hypothesis is that surface-level patterns (naming, comments) are more discriminative than data flow for this task." |
| "Why not larger dataset?" | "We used 30k samples. CodeBERT saturates quickly due to pre-training — diminishing returns after ~50k samples for fine-tuning." |
| "Why not ensemble?" | "Time constraint. Future work could combine CodeBERT with perplexity-based features." |

---

## Code Templates

### TF-IDF Baseline (baseline_tfidf.py)
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import pandas as pd

def train_tfidf_baseline(train_df, val_df, task="A"):
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(train_df['code'])
    X_val = vectorizer.transform(val_df['code'])
    
    clf = LogisticRegression(max_iter=1000, class_weight='balanced')
    clf.fit(X_train, train_df['label'])
    
    preds = clf.predict(X_val)
    
    if task == "A":
        f1 = f1_score(val_df['label'], preds, average='binary')
    else:
        f1 = f1_score(val_df['label'], preds, average='macro')
    
    return f1, clf, vectorizer
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
        print(f"TRUE: {row['generator']} (label={row['label']})")
        print(f"PRED: label={row['pred']}")
        print(f"LANGUAGE: {row['language']}")
        print("-" * 40)
        print(row['code'][:500])  # First 500 chars
        print("=" * 80)
        input("Press Enter for next sample...")
```

---

## Communication Protocol

| Time | Sync Action |
|------|-------------|
| End of Day 1 | Share: All 4 F1 numbers (TF-IDF A/B, CodeBERT A/B) |
| Mid Day 2 | Share: Ablation results, start error analysis together |
| End of Day 2 | Share: All PNGs, error analysis notes |
| Mid Day 3 | Review poster draft together, practice Q&A |

---

## Key Resources

- **Dataset:** [HuggingFace](https://huggingface.co/datasets/DaniilOr/SemEval-2026-Task13)
- **GitHub:** [mbzuai-nlp/SemEval-2026-Task13](https://github.com/mbzuai-nlp/SemEval-2026-Task13)
- **Kaggle Task A:** [Competition Link](https://www.kaggle.com/t/99673e23fe8546cf9a07a40f36f2cc7e)
- **Kaggle Task B:** [Competition Link](https://www.kaggle.com/t/65af9e22be6d43d884cfd6e41cad3ee4)
- **CodeBERT:** [microsoft/codebert-base](https://huggingface.co/microsoft/codebert-base)
- **CodeBERT Paper:** [arXiv:2002.08155](https://arxiv.org/abs/2002.08155)

---

## Academic Narrative (For Poster/Presentation)

> **Story arc:**
> 
> 1. **Problem:** LLMs generate increasingly human-like code, threatening academic integrity and software supply chain security.
> 
> 2. **Challenge:** Detection must generalize across languages (Python, Java, C++, Go, PHP) and generator families (10 LLM types).
> 
> 3. **Approach:** We compare TF-IDF (lexical baseline) with CodeBERT (semantic understanding) to quantify the importance of code semantics.
> 
> 4. **Findings:** 
>    - CodeBERT outperforms TF-IDF by ___%, confirming semantic features matter
>    - Model struggles with [language] and confuses [generator pairs]
>    - Preprocessing [helps/hurts] by ___% — [insight]
> 
> 5. **Conclusion:** Pre-trained code models are effective but limited by training language distribution. Future work should explore cross-lingual transfer and stylometric features.

This is a **complete, defensible academic story** — not just "we got good numbers."