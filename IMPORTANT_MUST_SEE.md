# NLP Project Implementation Timeline
## SemEval-2026 Task 13: Detecting Machine-Generated Code
**Team 32:** Anand Karna & Zeev Tayer  
**Deadline:** January 24, 2026 (Poster Submission)

---

## Overview

| Day | Focus | End Goal |
|-----|-------|----------|
| Day 1 | Data + Working Baseline | Two F1 numbers (Task A, Task B) |
| Day 2 | Improve + Visualizations | All figures, Kaggle submission |
| Day 3 | Poster | Final `poster.pdf` (A0) |

---

## DAY 1: Data + Working Baseline

| # | Task | Owner | Done When | Time Est |
|---|------|-------|-----------|----------|
| 1 | Download dataset from HuggingFace (`DaniilOr/SemEval-2026-Task13`) | Either | `datasets` loads Task A and B train/val splits | 15 min |
| 2 | Create 15k stratified subset for Task A | Anand | File `task_a_subset.parquet` exists, ~7.5k per class | 30 min |
| 3 | Create 15k stratified subset for Task B | Zeev | File `task_b_subset.parquet` exists, proportional 11 classes | 30 min |
| 4 | Write shared preprocessing function (remove comments, normalize whitespace) | Anand | Function runs on 100 samples, outputs clean strings | 1 hr |
| 5 | Set up CodeBERT tokenizer + data loader | Anand | Can tokenize batch of 32, tensors shape `(32, 512)` | 1 hr |
| 6 | Implement Task A training script (binary classification) | Anand | `train_task_a.py` runs 1 epoch without crash | 2 hr |
| 7 | Implement Task B training script (11-class) | Zeev | `train_task_b.py` runs 1 epoch without crash | 2 hr |
| 8 | Train Task A on 15k subset (3 epochs) | Anand | Loss decreases, model checkpoint saved | 1-2 hr |
| 9 | Train Task B on 15k subset (3 epochs) | Zeev | Loss decreases, model checkpoint saved | 1-2 hr |
| 10 | Evaluate both on validation set | Both | Two F1 numbers printed (Task A F1, Task B Macro F1) | 30 min |

### ✅ Day 1 Checkpoint
> **You can say:** "Task A F1 = ___, Task B Macro F1 = ___" with actual numbers.

---

## DAY 2: Improve + Generate All Visuals

| # | Task | Owner | Done When | Time Est |
|---|------|-------|-----------|----------|
| 11 | Hyperparameter sweep Task A (lr: 1e-5, 2e-5, 5e-5) | Anand | Table with 3 rows: lr → F1 | 2 hr |
| 12 | Hyperparameter sweep Task B + implement Focal Loss | Zeev | Table with 3 rows: lr → Macro F1 | 2 hr |
| 13 | Train final Task A model on 30k subset with best lr | Anand | `model_task_a.pt` saved | 1-2 hr |
| 14 | Train final Task B model on 30k subset with best lr | Zeev | `model_task_b.pt` saved | 1-2 hr |
| 15 | Generate Task A confusion matrix | Anand | `confusion_matrix_task_a.png` exists | 30 min |
| 16 | Generate Task B confusion matrix (11×11) | Zeev | `confusion_matrix_task_b.png` exists | 30 min |
| 17 | Generate training loss curves (both tasks) | Either | `loss_curve_a.png`, `loss_curve_b.png` exist | 30 min |
| 18 | Calculate per-language F1 for Task A | Anand | Table: Python/Java/C++ → F1 | 30 min |
| 19 | Calculate per-generator F1 for Task B | Zeev | Table: 11 generators → F1 | 30 min |
| 20 | Identify top 3 confusion pairs in Task B | Zeev | Written: "Model confuses X with Y" | 30 min |
| 21 | Generate Kaggle submission files | Both | `submission_task_a.csv`, `submission_task_b.csv` | 30 min |
| 22 | Submit to Kaggle (get leaderboard score) | Both | Kaggle shows your score | 15 min |

### ✅ Day 2 Checkpoint
> **All PNGs exist, all tables filled, Kaggle scores recorded.**

---

## DAY 3: Poster

| # | Task | Owner | Done When | Time Est |
|---|------|-------|-----------|----------|
| 23 | Set up A0 LaTeX poster template | Anand | `poster.tex` compiles to empty A0 PDF | 1 hr |
| 24 | Write Introduction + Problem Statement (~150 words) | Anand | Text in poster, readable | 1 hr |
| 25 | Write Methodology section + architecture diagram | Zeev | CodeBERT diagram + 100 words in poster | 1 hr |
| 26 | Insert all figures (confusion matrices, loss curves, distributions) | Anand | Images render correctly in PDF | 1 hr |
| 27 | Write Results section with F1 tables | Zeev | Tables embedded in poster | 1 hr |
| 28 | Write Analysis + Key Findings (3 insights) | Zeev | Bullet points in poster | 30 min |
| 29 | Write Conclusion + Future Work (~100 words) | Anand | Text in poster | 30 min |
| 30 | Add References (CodeBERT paper, Task13 papers) | Either | BibTeX renders correctly | 15 min |
| 31 | Final review: font sizes, layout, names visible | Both | Text readable from 1 meter | 1 hr |
| 32 | Export final `poster.pdf` | Both | A0 PDF, < 20MB, both names on it | 15 min |

### ✅ Day 3 Checkpoint
> **`poster.pdf` exists, is A0 size, has "Anand Karna" and "Zeev Tayer" visible.**

---

## Project Structure

```
project/
├── data/
│   ├── task_a_subset.parquet       # Step 2
│   └── task_b_subset.parquet       # Step 3
├── src/
│   ├── preprocess.py               # Step 4
│   ├── train_task_a.py             # Step 6
│   ├── train_task_b.py             # Step 7
│   └── evaluate.py                 # Step 10
├── outputs/
│   ├── models/
│   │   ├── model_task_a.pt         # Step 13
│   │   └── model_task_b.pt         # Step 14
│   ├── figures/
│   │   ├── confusion_matrix_task_a.png  # Step 15
│   │   ├── confusion_matrix_task_b.png  # Step 16
│   │   ├── loss_curve_a.png        # Step 17
│   │   └── loss_curve_b.png        # Step 17
│   └── submissions/
│       ├── submission_task_a.csv   # Step 21
│       └── submission_task_b.csv   # Step 21
└── poster/
    ├── poster.tex                  # Step 23
    └── poster.pdf                  # Step 32
```

---

## Quick Reference Commands

```python
# Load data (Step 1)
from datasets import load_dataset
ds_a = load_dataset("DaniilOr/SemEval-2026-Task13", "A")  # Task A
ds_b = load_dataset("DaniilOr/SemEval-2026-Task13", "B")  # Task B

# Model setup (Steps 6-7)
from transformers import AutoModelForSequenceClassification, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model_a = AutoModelForSequenceClassification.from_pretrained(
    "microsoft/codebert-base", num_labels=2
)  # Task A: Binary
model_b = AutoModelForSequenceClassification.from_pretrained(
    "microsoft/codebert-base", num_labels=11
)  # Task B: 11-class
```

---

## Communication Protocol

| Time | Sync Action |
|------|-------------|
| End of Day 1 | Share: F1 numbers, any blockers |
| Mid Day 2 | Share: Best hyperparameters found |
| End of Day 2 | Share: All figures as PNGs |
| Mid Day 3 | Review poster draft together |

---

## Key Resources

- **Dataset:** [HuggingFace](https://huggingface.co/datasets/DaniilOr/SemEval-2026-Task13)
- **GitHub:** [mbzuai-nlp/SemEval-2026-Task13](https://github.com/mbzuai-nlp/SemEval-2026-Task13)
- **Kaggle Task A:** [Competition Link](https://www.kaggle.com/t/99673e23fe8546cf9a07a40f36f2cc7e)
- **Kaggle Task B:** [Competition Link](https://www.kaggle.com/t/65af9e22be6d43d884cfd6e41cad3ee4)
- **CodeBERT:** [microsoft/codebert-base](https://huggingface.co/microsoft/codebert-base)

---

## Progress Tracker

### Day 1
- [ ] Step 1: Download dataset
- [ ] Step 2: Task A subset
- [ ] Step 3: Task B subset
- [ ] Step 4: Preprocessing function
- [ ] Step 5: Tokenizer + data loader
- [ ] Step 6: Task A training script
- [ ] Step 7: Task B training script
- [ ] Step 8: Train Task A
- [ ] Step 9: Train Task B
- [ ] Step 10: Evaluate both

### Day 2
- [ ] Step 11: HP sweep Task A
- [ ] Step 12: HP sweep Task B + Focal Loss
- [ ] Step 13: Final Task A model
- [ ] Step 14: Final Task B model
- [ ] Step 15: Confusion matrix A
- [ ] Step 16: Confusion matrix B
- [ ] Step 17: Loss curves
- [ ] Step 18: Per-language F1
- [ ] Step 19: Per-generator F1
- [ ] Step 20: Top 3 confusion pairs
- [ ] Step 21: Kaggle submissions
- [ ] Step 22: Submit to Kaggle

### Day 3
- [ ] Step 23: LaTeX template
- [ ] Step 24: Introduction
- [ ] Step 25: Methodology
- [ ] Step 26: Insert figures
- [ ] Step 27: Results section
- [ ] Step 28: Analysis
- [ ] Step 29: Conclusion
- [ ] Step 30: References
- [ ] Step 31: Final review
- [ ] Step 32: Export PDF

---

## Results (Fill in as you go)

### Task A (Binary Classification)
| Metric | Value |
|--------|-------|
| Validation F1 | ___ |
| Kaggle Score | ___ |
| Best Learning Rate | ___ |

### Task B (11-class)
| Metric | Value |
|--------|-------|
| Validation Macro F1 | ___ |
| Kaggle Score | ___ |
| Best Learning Rate | ___ |

### Per-Language F1 (Task A)
| Language | F1 |
|----------|-----|
| Python | ___ |
| Java | ___ |
| C++ | ___ |

### Per-Generator F1 (Task B)
| Generator | F1 |
|-----------|-----|
| Human | ___ |
| DeepSeek-AI | ___ |
| Qwen | ___ |
| 01-ai | ___ |
| BigCode | ___ |
| Gemma | ___ |
| Phi | ___ |
| Meta-LLaMA | ___ |
| IBM-Granite | ___ |
| Mistral | ___ |
| OpenAI | ___ |