# Feature Specification: Machine-Generated Code Detection (Zeev's Tasks)

**Feature Branch**: `002-zeev-mgc-detection`
**Created**: 2026-01-22
**Status**: Active
**Deadline**: January 24, 2026 (Poster Submission)
**Task**: SemEval-2026 Task 13 - Detecting Machine-Generated Code (Task B)

---

## Overview

This specification covers **Zeev Tayer's individual contributions** for Team 32's SemEval-2026 Task 13 project. The scope is Task B: multiclass per-generator classification (11 classes), including baseline implementation, CodeBERT training for Task B, per-generator analysis, confusion matrices, and poster figures.

**Scope**: Task B only (11-class generator classification), per-generator evaluation and visualization, CodeBERT training pipeline for Task B, TF-IDF baseline for Task B, poster materials owned by Zeev.

---

## User Scenarios & Goals

### Goal 1 - Data Preparation & Baseline (Priority: P1, Day 1)

Prepare the dataset and establish a TF-IDF baseline for Task B.

**Acceptance Criteria**:

1. Given the HuggingFace dataset, when download and subset script runs, then `task_b_subset.parquet` exists with ~20k stratified samples (stratify by label+language).
2. Given the subset data, when TF-IDF + Logistic Regression trains, then Macro F1 score is printed (expected baseline ~0.40-0.60 depending on class balance).
3. Given raw code samples, when preprocessing function runs, then comments are removed and whitespace is normalized (for ablation experiments).
4. Given the preprocessing function, when tested on 100 samples, then all samples transform without errors and both `code` and `code_preprocessed` columns are stored.

---

### Goal 2 - CodeBERT Implementation (Priority: P1, Day 1)

Implement and train CodeBERT for Task B to establish a transformer baseline.

**Acceptance Criteria**:

1. Given CodeBERT tokenizer, when a batch of 32 samples is tokenized, then output shape is `(32, 512)`.
2. Given `train_task_b.py`, when 1 epoch runs, then script completes without crash (smoke test).
3. Given training for 3 epochs on 20k samples, when training completes, then loss decreases and a checkpoint `model_task_b.pt` is saved.
4. Given trained model, when evaluated on validation set, then Macro F1 is reported and per-generator F1 table is saved.

---

### Goal 3 - Ablation & Analysis (Priority: P1, Day 2)

Run ablation (raw vs preprocessed) and generate analysis artifacts for the poster.

**Acceptance Criteria**:

1. Given CodeBERT model, when trained WITH and WITHOUT preprocessing, then a table comparing raw vs preprocessed Macro F1 exists.
2. Given validation predictions, when per-generator analysis runs, then `per_generator_f1.csv` (11 rows) and a PNG table are saved.
3. Given predictions, when confusion matrix is generated, then `confusion_matrix_task_b.png` (11×11) and `confusion_matrix_task_b.npy` are saved.
4. Given analysis results, when top-3 confusion pairs are identified, then `notes/top3_confusions.md` contains short explanations.

---

### Goal 4 - Poster Creation (Priority: P1, Day 3)

Create A0 poster sections owned by Zeev: methodology diagram for Task B, per-generator F1 table, confusion matrix, and error-analysis notes.

**Acceptance Criteria**:

1. Given poster template, when figures are inserted, then `poster.pdf` includes per-generator tables and confusion matrix figures.
2. Given error analysis, when top-3 confusions are summarized, then a one-paragraph insight is present in the poster.

---

## Requirements

### Functional Requirements (Zeev - Task B)

- **FR-Z01**: System MUST download SemEval-2026 Task 13 dataset from HuggingFace.
- **FR-Z02**: System MUST create ~20k stratified subset for Task B and save as `data/task_b_subset.parquet`.
- **FR-Z03**: System MUST implement TF-IDF + Logistic Regression baseline for Task B with max_features=10000, ngram_range=(1,3), min_df=2 and report Macro F1.
- **FR-Z04**: System MUST implement preprocessing function (comment removal, whitespace normalization) and store `code_preprocessed` alongside `code`.
- **FR-Z05**: System MUST implement CodeBERT fine-tuning script `train_task_b.py` for multiclass classification (11 classes) with `--preprocess` flag.
- **FR-Z06**: System MUST run a 1-epoch smoke test for `train_task_b.py` without crash and save a small checkpoint.
- **FR-Z07**: System MUST compute per-generator Macro F1 and save `outputs/002-zeev-mgc-detection/figures/per_generator_f1.csv`.
- **FR-Z08**: System MUST save confusion matrix PNG and `.npy` to `outputs/002-zeev-mgc-detection/figures/`.
- **FR-Z09**: System MUST save top-3 confusion notes to `outputs/002-zeev-mgc-detection/notes/top3_confusions.md`.
- **FR-Z10**: System MUST save logs, stdout prints, metrics, and figures under `outputs/002-zeev-mgc-detection/`.

### Technical Constraints

- Python 3.9+ with PyTorch and transformers
- CodeBERT model: `microsoft/codebert-base` (first 512 tokens default)
- Dataset: `DaniilOr/SemEval-2026-Task13`
- Parquet format for data storage
- GPU training recommended for full runs; support CPU/MPS for smoke tests

---

## Success Criteria

| Metric                 | Target        | Rationale                                 |
| ---------------------- | ------------- | ----------------------------------------- |
| TF-IDF Macro F1        | 0.40-0.60     | Lexical baseline (expected)               |
| CodeBERT Macro F1      | 0.65+         | Transformer improvement over TF-IDF       |
| Per-generator table    | 11 rows       | All generator classes reported            |
| Confusion matrix saved | PNG + .npy    | For poster and reproducibility            |
| Poster artifacts exist | Yes           | Figures + short analysis included         |

---

## Key Outputs

### Data Artifacts

- `data/task_b_subset.parquet` - ~20k stratified samples (label + language)

### Model Artifacts

- `outputs/002-zeev-mgc-detection/models/model_task_b.pt` - small checkpoint from smoke test; final model saved after full training

### Visualization Artifacts

- `outputs/002-zeev-mgc-detection/figures/confusion_matrix_task_b.png`
- `outputs/002-zeev-mgc-detection/figures/confusion_matrix_task_b.npy`
- `outputs/002-zeev-mgc-detection/figures/per_generator_f1.csv` and PNG

### Submission Artifacts

- `outputs/002-zeev-mgc-detection/submissions/submission_task_b.csv`

### Documentation

- `spec.md`, `research.md`, `plan.md`, `tasks.md` (this folder)
- `src/train_task_b.py` (training script)

---

## Assumptions

- Dataset is publicly available on HuggingFace and accessible from this environment.
- GPU access is available for full runs; smoke tests may run on CPU/MPS.
- Primary experiment uses raw `code`; `--preprocess` flag enables ablation.

---

## Notes & Next Steps (Day 1)

1. Download dataset and create `task_b_subset.parquet` (Zeev).
2. Implement TF-IDF baseline for Task B and print Macro F1 (Zeev).
3. Implement `train_task_b.py` smoke test (Zeev).
4. Produce per-generator F1 table and confusion matrix (Zeev).

