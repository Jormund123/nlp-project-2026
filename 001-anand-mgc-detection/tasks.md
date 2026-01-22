# Tasks: Machine-Generated Code Detection (Anand)

**Input**: Design documents from `002-anand-mgc-detection/`  
**Prerequisites**: spec.md, plan.md, research.md  
**Deadline**: January 24, 2026

---

## Format: `[ID] [Day] [P?] Description`

- **[P]**: Can run in parallel (no dependencies)
- **[Day]**: Which day this task belongs to (Day 1, Day 2, Day 3)

---

## Phase 1: Environment & Data Setup (Day 1) ⏳

**Purpose**: Set up development environment and prepare dataset

- [x] T001 [Day1] Create/verify virtual environment with all dependencies from requirements.txt
- [x] T002 [Day1] Verify GPU access (torch.cuda.is_available() returns True) - Device available MPS
- [x] T003 [Day1] Download SemEval-2026 Task 13 dataset from HuggingFace
- [x] T004 [Day1] Create 20k stratified subset for Task A → `data/task_a_subset.parquet`
  - ⚠️ **Critical**: Stratify by BOTH label AND language (not just label)
  - Create combined stratification column: `df['stratify_col'] = df['label'].astype(str) + "_" + df['language']`
- [x] T005 [Day1] Verify stratification: print language distribution in train AND val sets
- [x] T006 [Day1] Implement `remove_comments()` function in `src/preprocess.py`
- [x] T007 [Day1] Implement `normalize_whitespace()` function in `src/preprocess.py`
- [x] T008 [Day1] Implement `preprocess_code()` wrapper in `src/preprocess.py`
- [x] T009 [Day1] Add both `code` (raw) and `code_preprocessed` columns to parquet
- [x] T010 [Day1] Test preprocessing on 100 samples (verify no crashes)

**Checkpoint**: ✅ `task_a_subset.parquet` exists with 20k samples, language distribution balanced between train/val

---

## Phase 2: TF-IDF Baseline (Day 1) ⏳

**Purpose**: Establish lexical baseline for comparison

- [x] T011 [Day1] Implement `train_tfidf_baseline()` in `src/baseline_tfidf.py`
  - Config: `max_features=10000, ngram_range=(1, 3), min_df=2`
  - Use `class_weight='balanced'` for LogisticRegression
- [x] T012 [Day1] Train TF-IDF + LogReg on Task A subset (use raw `code` column)
- [x] T013 [Day1] Evaluate and print F1 score (expect ~0.60-0.70)
- [x] T014 [Day1] Save TF-IDF model and vectorizer for later use

**Checkpoint**: ✅ TF-IDF F1 score recorded: **0.8310** (ROC-AUC: 0.9107)

---

## Phase 3: CodeBERT Implementation (Day 1) ⏳

**Purpose**: Implement transformer-based classifier

- [x] T015 [Day1] Set up CodeBERT tokenizer (`microsoft/codebert-base`)
- [x] T016 [Day1] Implement `CodeDataset` class in `src/train_task_a.py`
- [x] T017 [Day1] Verify batch tokenization shape `(32, 512)`
- [x] T018 [Day1] Implement `CodeBERTClassifier` model class (768 → 2 classification head)
- [x] T019 [Day1] Implement training loop with AdamW optimizer
  - Config: `lr=2e-5, weight_decay=0.01`
- [x] T020 [Day1] Implement learning rate scheduler (linear warmup, `warmup_ratio=0.1`)
- [x] T021 [Day1] Test training script runs 1 epoch without crash
- [x] T022 [Day1] Train full model on RAW code (3 epochs, 20k samples)
  - ⚠️ **Primary experiment**: Use raw `code` column (preserves AI fingerprints)
- [x] T023 [Day1] Save checkpoint to `outputs/models/model_task_a_raw.pt`
- [x] T024 [Day1] Record training losses for each epoch: 0.1819, 0.0532, 0.0326

**Checkpoint**: ✅ CodeBERT F1 score recorded: **0.9854** (ROC-AUC: 0.9988)

---

## Phase 4: Evaluation & Validation (Day 1/2) ⏳

**Purpose**: Evaluate both models on validation set

- [x] T025 [Day1] Implement `evaluate_model()` function in `src/evaluate.py`
  - Return: F1, precision, recall, confusion_matrix, ROC-AUC (if proba provided)
- [x] T026 [Day1] Evaluate TF-IDF on validation set → record F1
- [x] T027 [Day1] Evaluate CodeBERT on validation set → record F1, ROC-AUC
- [x] T028 [Day1] Fill baseline comparison table:

| Model           | Task A F1 | ROC-AUC            |
| --------------- | --------- | ------------------ |
| TF-IDF + LogReg | 0.8310    | 0.9107             |
| CodeBERT (raw)  | 0.9854    | 0.9988             |
| **Improvement** | +18.58%   | (Significant jump) |

**Checkpoint**: ✅ Both F1 scores recorded, improvement calculated

---

## Phase 5: Ablation Study (Day 2) ⏳

**Purpose**: Test the "Fingerprint Paradox" hypothesis

> **Hypothesis**: Raw code will outperform preprocessed code because AI signatures live in comments, formatting, and whitespace patterns.

- [ ] T029 [Day2] Add `--preprocess` flag to training script (default is RAW code)
- [ ] T030 [Day2] Train CodeBERT WITH preprocessing (`code_preprocessed` column)
- [ ] T031 [Day2] Save checkpoint to `outputs/models/model_task_a_preprocessed.pt`
- [ ] T032 [Day2] Evaluate preprocessed model on validation set
- [ ] T033 [Day2] Compare F1: raw vs preprocessed
- [ ] T034 [Day2] Fill ablation table:

| Variant                 | Task A F1 | Notes                           |
| ----------------------- | --------- | ------------------------------- |
| **Raw code** (primary)  | \_\_\_    | Preserves AI fingerprints       |
| Preprocessed (ablation) | \_\_\_    | Removes discriminative features |
| **Difference**          | ±\_\_\_%  |                                 |

- [ ] T035 [Day2] Write 1-sentence insight about preprocessing impact (confirm/reject hypothesis)

**Checkpoint**: ✅ Ablation results documented, hypothesis tested

---

## Phase 6: Per-Language Analysis (Day 2) ⏳

**Purpose**: Analyze model performance across programming languages

- [ ] T036 [Day2] Implement `compute_per_language_f1()` in `src/evaluate.py`
- [ ] T037 [Day2] Compute F1 for Python samples
- [ ] T038 [Day2] Compute F1 for Java samples
- [ ] T039 [Day2] Compute F1 for C++ samples
- [ ] T040 [Day2] Fill per-language table:

| Language | Samples | F1     | Notes                                    |
| -------- | ------- | ------ | ---------------------------------------- |
| Python   | \_\_\_  | \_\_\_ | Expected: hardest (most human variation) |
| Java     | \_\_\_  | \_\_\_ |                                          |
| C++      | \_\_\_  | \_\_\_ |                                          |

**Checkpoint**: ✅ Per-language breakdown complete

---

## Phase 7: Visualization Generation (Day 2) ⏳

**Purpose**: Create all figures for poster (academic standard, 300 DPI)

**Style Config**:

```python
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {'tfidf': '#1f77b4', 'codebert': '#ff7f0e', 'raw': '#2ca02c', 'preprocessed': '#7f7f7f'}
```

- [ ] T041 [Day2] [P] Implement `generate_confusion_matrix()` in `src/evaluate.py`
- [ ] T042 [Day2] [P] Generate `outputs/figures/confusion_matrix_task_a.png`
- [ ] T043 [Day2] [P] Implement `generate_loss_curve()` in `src/evaluate.py`
- [ ] T044 [Day2] [P] Generate `outputs/figures/loss_curve_a.png`
- [ ] T045 [Day2] [P] Implement `generate_baseline_comparison()` in `src/evaluate.py`
- [ ] T046 [Day2] [P] Generate `outputs/figures/baseline_comparison.png`
- [ ] T047 [Day2] [P] Implement `generate_pr_curve()` in `src/evaluate.py`
- [ ] T048 [Day2] [P] Generate `outputs/figures/pr_curve_task_a.png`

**Checkpoint**: ✅ All 4 PNG files exist (confusion matrix, loss curve, baseline comparison, PR curve)

---

## Phase 8: Kaggle Submission (Day 2) ⏳

**Purpose**: Generate and submit predictions to Kaggle

- [ ] T049 [Day2] Load test set from HuggingFace dataset
- [ ] T050 [Day2] Generate predictions using best model (raw code CodeBERT)
- [ ] T051 [Day2] Create `outputs/submissions/submission_task_a.csv`
- [ ] T052 [Day2] Submit to Kaggle Task A competition
- [ ] T053 [Day2] Record leaderboard score: \_\_\_

**Checkpoint**: ✅ Kaggle submission complete

---

## Phase 9: Poster Creation (Day 3) ⏳

**Purpose**: Create A0 academic poster using baposter class

- [ ] T054 [Day3] Set up baposter LaTeX template (3-column A0 portrait)
- [ ] T055 [Day3] Verify template compiles to A0 PDF
- [ ] T056 [Day3] Add title: "Detecting Machine-Generated Code with CodeBERT"
- [ ] T057 [Day3] Add names: Team 32 - Anand Karna & Zeev Tayer
- [ ] T058 [Day3] Write Introduction section (~150 words)
- [ ] T059 [Day3] Insert baseline comparison figure
- [ ] T060 [Day3] Insert results tables (baseline + per-language)
- [ ] T061 [Day3] Write Ablation Results section (table + "Fingerprint Paradox" insight)
- [ ] T062 [Day3] Insert PR curve figure
- [ ] T063 [Day3] Write Conclusion section (~100 words)
- [ ] T064 [Day3] Add References (CodeBERT paper, Task13 dataset)
- [ ] T065 [Day3] Final layout review (font sizes, readability from 1m)
- [ ] T066 [Day3] Export final `poster/poster.pdf` (<20MB)

**Checkpoint**: ✅ `poster.pdf` exists with all required sections

---

## Task Count Summary

| Phase                    | Tasks    | Time Est. |
| ------------------------ | -------- | --------- |
| Phase 1 (Setup)          | 10 tasks | 1.5 hr    |
| Phase 2 (TF-IDF)         | 4 tasks  | 1 hr      |
| Phase 3 (CodeBERT)       | 10 tasks | 4 hr      |
| Phase 4 (Evaluation)     | 4 tasks  | 0.5 hr    |
| Phase 5 (Ablation)       | 7 tasks  | 2 hr      |
| Phase 6 (Per-Language)   | 5 tasks  | 0.5 hr    |
| Phase 7 (Visualizations) | 8 tasks  | 1 hr      |
| Phase 8 (Kaggle)         | 5 tasks  | 0.5 hr    |
| Phase 9 (Poster)         | 13 tasks | 4 hr      |

**Total**: 66 tasks, ~15 hours

---

## Critical Path

```
T001-T010 (Setup/Data + Stratification Verification)
    ↓
T011-T014 (TF-IDF) ←→ T015-T024 (CodeBERT on RAW) [parallel after data ready]
    ↓                      ↓
T025-T028 (Evaluation) ←───┘
    ↓
T029-T035 (Ablation: preprocessed)
    ↓
T036-T053 (Analysis + Visualizations + Kaggle) [lots of parallel]
    ↓
T054-T066 (Poster)
```

---

## Key Changes from Research

| Area               | Before            | After                   |
| ------------------ | ----------------- | ----------------------- |
| Stratification     | Label only        | **Label + Language**    |
| Primary experiment | Preprocessed      | **Raw code**            |
| Ablation flag      | `--no-preprocess` | `--preprocess`          |
| TF-IDF n-grams     | (1, 2)            | **(1, 3)**              |
| Figures            | 3 PNGs            | **4 PNGs** (+ PR curve) |
| Metrics            | F1 only           | **F1 + ROC-AUC**        |

---

## Hypotheses to Validate

| Hypothesis              | Task      | Expected                   |
| ----------------------- | --------- | -------------------------- |
| Raw code > preprocessed | T033-T035 | Raw code has higher F1     |
| CodeBERT > TF-IDF       | T028      | 15-20% F1 improvement      |
| Python hardest          | T037-T040 | Lowest F1 (most variation) |

---

## Notes

- [P] tasks can run in parallel
- Phases 1-4 are **blocking** - must complete on Day 1
- Phases 5-8 have parallel opportunities on Day 2
- Phase 9 is **blocking** - must complete on Day 3
- Coordinate with Zeev for Task B figures (confusion matrix, per-generator table)
- ⚠️ Always use raw `code` column for primary experiments
