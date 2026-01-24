# Tasks: Machine-Generated Code Detection (Zeev)

## Format: `[ID] [Day] [P?] Description` — Zeev tasks

## Phase 1: Environment & EDA (Day 1)

- Z001 [Day1] Verify HuggingFace dataset access and download small sample for smoke tests (owner: Zeev) — prints saved to `outputs/002-zeev-mgc-detection/logs/`.
- Z002 [Day1] Exploratory data analysis: class distribution (per-generator), language counts, code length stats — save `eda_summary.csv` and `eda_summary.md` (30 min).
- Z003 [Day1] Create Task B stratified subset (n ≈ 20k or as available) preserving generator label distribution — save `data/task_b_subset.parquet` (30 min).

## Phase 2: TF-IDF Baseline (Day 1)

- Z004 [Day1] Implement `train_tfidf_baseline_b()` in `src/baseline_tfidf.py` or `src/baseline_tfidf_b.py` — config: max_features=10000, ngram_range=(1,3) (1 hr).
- Z005 [Day1] Train TF-IDF + Logistic Regression on Task B subset (print Macro F1) — save model and vectorizer to `outputs/002-zeev-mgc-detection/models/tfidf_task_b.pkl` (1 hr).

## Phase 3: CodeBERT Implementation (Day 1)

- Z006 [Day1] Setup tokenizer (`microsoft/codebert-base`) and implement `CodeDataset` and multiclass DataLoader for Task B (1 hr).
- Z007 [Day1] Implement `train_task_b.py` (multiclass head, argparse with `--preprocess`, saving checkpoints) (2 hr).
- Z008 [Day1] Smoke test: run `train_task_b.py` for 1 epoch on 1000 samples, ensure no crash and save small checkpoint (1 hr).

## Phase 4: Evaluation & Analysis (Day 2)

- Z009 [Day2] Implement `evaluate_model_b()` in `src/evaluate.py` to compute Macro F1, per-generator F1, confusion matrix — save CSV and PNG (1 hr).
- Z010 [Day2] Compute per-generator F1 table and save `outputs/002-zeev-mgc-detection/figures/per_generator_f1.csv` (30 min).
- Z011 [Day2] Generate normalized and absolute confusion matrix PNG `confusion_matrix_task_b.png` and `.npy` (20 min).
- Z012 [Day2] Identify top-3 confusion pairs and save `outputs/002-zeev-mgc-detection/notes/top3_confusions.md` (30 min).

## Phase 5: Poster & Figures (Day 2-3)

- Z013 [Day2] Prepare `per_generator_f1.png` for poster (style-consistent) (30 min).
- Z014 [Day3] Prepare `loss_curve_b.png` and `baseline_comparison.png` contributions if applicable (30 min).

# Tasks: Machine-Generated Code Detection (Zeev)

**Input**: Design documents from `002-zeev-mgc-detection/`  
**Prerequisites**: spec.md, plan.md, research.md  
**Deadline**: January 24, 2026

---

## Format: `[ID] [Day] [P?] Description`

- **[P]**: Can run in parallel (no dependencies)
- **[Day]**: Which day this task belongs to (Day 1, Day 2, Day 3)

---

## Phase 1: Environment & Data Setup (Day 1) ⏳

**Purpose**: Set up development environment and prepare Task B dataset

- [x] T001 [Day1] Create/verify virtual environment with all dependencies (requirements.txt / environment.yml)
- [x] T002 [Day1] Verify device availability (torch.cuda.is_available() or MPS)
- [x] T003 [Day1] Download SemEval-2026 Task 13 dataset from HuggingFace
- [x] T004 [Day1] Create ~20k stratified subset for Task B → `data/task_b_subset.parquet`
  - ⚠️ **Critical**: Stratify by BOTH label AND language (not just label)
  - Create combined stratification column: `df['stratify_col'] = df['label'].astype(str) + "_" + df['language']`
- [x] T005 [Day1] Verify stratification: print language distribution in train AND val sets (distributions within ~2%)
- [x] T006 [Day1] Implement `remove_comments()` and `normalize_whitespace()` in `src/preprocess.py`
- [x] T007 [Day1] Implement `preprocess_code()` wrapper and add `code_preprocessed` column
- [x] T008 [Day1] Test preprocessing on 100 samples (verify no crashes)

**Checkpoint**: ✅ `task_b_subset.parquet` exists with ~20k samples, stratification verified

---

## Phase 2: TF-IDF Baseline (Day 1) ✅

**Purpose**: Establish lexical baseline for Task B

- [x] T009 [Day1] Implement `train_tfidf_baseline()` in `src/baseline_tfidf_b.py`
  - Config: `max_features=10000, ngram_range=(1,3), min_df=2`
  - Use `class_weight='balanced'` for LogisticRegression
  - Persist artifacts: `outputs/002-zeev-mgc-detection/models/tfidf_task_b.pkl`
- [x] T010 [Day1] Train TF-IDF + LogReg on Task B subset (use raw `code` column)
- [x] T011 [Day1] Evaluate and print Macro F1 (expected ~0.40-0.60) -> Actual: **0.4523**
- [x] T012 [Day1] Save TF-IDF vectorizer and classifier to outputs

**Checkpoint**: ✅ TF-IDF Macro F1 recorded: **0.4523**

---

## Phase 3: CodeBERT Implementation (Day 1) ✅

**Purpose**: Implement transformer-based classifier for Task B

- [x] T013 [Day1] Set up CodeBERT tokenizer (`microsoft/codebert-base`) and config max_length=512
- [x] T014 [Day1] Implement `CodeDataset` class in `src/train_task_b.py` (handles `code` vs `code_preprocessed`)
- [x] T015 [Day1] Verify batch tokenization shape `(32, 512)` on a batch
- [x] T016 [Day1] Implement `CodeBERTClassifier` model class for multiclass (768 → 11)
- [x] T017 [Day1] Implement training loop with AdamW optimizer
  - Config: `lr=2e-5, weight_decay=0.01, warmup_ratio=0.1`
- [x] T018 [Day1] Test `train_task_b.py` runs 1 epoch without crash (smoke test) and save small checkpoint `model_task_b_epoch1.pt`
- [x] T019 [Day1] (Optional) Run a short 1k-sample epoch to sanity-check loss decrease

**Checkpoint**: ✅ `train_task_b.py` smoke test completes

---

## Phase 4: Evaluation & Validation (Day 1/2) ⏳

**Purpose**: Evaluate models on validation set and create baseline comparisons

- [x] T020 [Day1] Implement `evaluate_model()` in `src/evaluate.py` (return Macro F1, precision, recall, confusion_matrix, per-class metrics)
- [x] T021 [Day1] Evaluate TF-IDF on validation set → record Macro F1, per-class metrics
- [x] T022 [Day1] Evaluate CodeBERT on validation set → record Macro F1, per-class metrics
  - **Comparison**: TF-IDF (**0.4523**) vs CodeBERT Raw (**0.7214**) → +26.9% improvement
- [ ] T023 [Day1] Save per-generator table `outputs/002-zeev-mgc-detection/figures/per_generator_f1.csv` (columns: generator, samples, precision, recall, f1)
- [x] T024 [Day1] Fill baseline comparison summary for poster

**Checkpoint**: ✅ Evaluation metrics comparison recorded

---

## Phase 5: Ablation Study (Day 2) ✅

**Purpose**: Test the "Fingerprint Paradox" (raw vs preprocessed)

- [x] T025 [Day2] Add `--preprocess` flag to `train_task_b.py` (default = raw)
- [x] T026 [Day2] Train CodeBERT WITH preprocessing and save checkpoint `model_task_b_preprocessed.pt`
- [x] T027 [Day2] Evaluate preprocessed model on validation set and compare Macro F1
- [x] T028 [Day2] Fill ablation table and write 1-sentence insight for poster:
      | Metric | Raw | Preprocessed | Change |
      | :--- | :--- | :--- | :--- |
      | Macro F1 | **0.7214** | 0.6855 | -3.59% |

  > **Insight**: Multi-class attribution relies more heavily on stylistic fingerprints than binary detection; removing them causes a larger drop (-3.6%) as generators become harder to distinguish.

**Checkpoint**: ✅ Ablation results documented

---

## Phase 6: Per-Generator & Per-Language Analysis (Day 2) ✅

**Purpose**: Produce fine-grained analysis required for poster

- [x] T029 [Day2] Compute per-generator Macro F1 and save CSV/PNG (11 rows)
- [x] T030 [Day2] Compute and save confusion matrix (raw counts + row-normalized) as PNG and `.npy`
- [x] T031 [Day2] Compute per-language Macro F1 breakdown and save table
- [x] T032 [Day2] Identify top-3 confusion pairs and write `notes/top3_confusions.md`

**Checkpoint**: ✅ Per-generator and per-language tables + notes saved

---

## Phase 7: Visualization Generation (Day 2) ✅

**Purpose**: Create poster-ready figures (300 DPI)

**Style Config**:

```python
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {'tfidf': '#1f77b4', 'codebert': '#ff7f0e', 'raw': '#2ca02c', 'preprocessed': '#7f7f7f'}
FIGURE_DPI = 300
```

- [x] T033 [Day2] Implement `generate_confusion_matrix()` and save `confusion_matrix_task_b.png` and `.npy`
- [x] T034 [Day2] Implement `plot_per_generator_f1()` and save PNG/table for poster
- [x] T035 [Day2] Generate training loss curves and save `loss_curve_b.png`
- [x] T036 [Day2] Generate baseline comparison figure and save `baseline_comparison_b.png`

**Checkpoint**: ✅ All poster figures saved under `outputs/002-zeev-mgc-detection/figures/`

---

## Phase 8: Poster & Writeups (Day 3) ⏳

**Purpose**: Assemble poster sections owned by Zeev and finalize visuals

- [ ] T037 [Day3] Insert per-generator table and confusion matrix into poster template (`poster/poster.tex`)
- [ ] T038 [Day3] Write Methods section (Task B: dataset, TF-IDF config, CodeBERT config)
- [ ] T039 [Day3] Write Analysis & Findings (top-3 confusions, per-language notes)
- [ ] T040 [Day3] Final layout review and export `poster/poster.pdf` (A0)

**Checkpoint**: ✅ Poster includes Task B figures and findings

---

## Task Count Summary (Zeev)

| Phase                    | Tasks | Time Est. |
| ------------------------ | ----- | --------- |
| Phase 1 (Setup)          | 8     | 1.5 hr    |
| Phase 2 (TF-IDF)         | 4     | 1 hr      |
| Phase 3 (CodeBERT)       | 7     | 3 hr      |
| Phase 4 (Evaluation)     | 5     | 0.5 hr    |
| Phase 5 (Ablation)       | 4     | 1.5 hr    |
| Phase 6 (Analysis)       | 4     | 1 hr      |
| Phase 7 (Visualizations) | 4     | 1 hr      |
| Phase 8 (Poster)         | 4     | 2 hr      |

**Total**: 40 tasks, ~11.5 hours

---

## Critical Path

```
T001-T008 (Setup/Data + Stratification Verification)
		↓
T009-T012 (TF-IDF)  ←→ T013-T019 (CodeBERT) [parallel after data ready]
		↓                      ↓
T020-T024 (Evaluation) ←───┘
		↓
T025-T028 (Ablation)
		↓
T029-T036 (Analysis + Visualizations)
		↓
T037-T040 (Poster)
```

---

## Notes

- [P] tasks can run in parallel where marked
- Day 1 is focused on data + baselines + smoke tests
- Day 2 is focused on ablation, analysis, and figure generation
- Day 3 is focused on poster assembly and final writeups
- Save all artifacts under `outputs/002-zeev-mgc-detection/` as specified in `spec.md`
