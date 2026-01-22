# Feature Specification: Machine-Generated Code Detection (Anand's Tasks)

**Feature Branch**: `002-anand-mgc-detection`  
**Created**: 2026-01-22  
**Status**: Active  
**Deadline**: January 24, 2026 (Poster Submission)  
**Task**: SemEval-2026 Task 13 - Detecting Machine-Generated Code

---

## Overview

This specification covers **Anand's individual contributions** to the NLP project for detecting machine-generated code. The project compares TF-IDF baseline with CodeBERT fine-tuning to demonstrate the importance of semantic understanding in code classification.

**Scope**: Task A only (Binary classification: Human vs. AI-generated code)

---

## User Scenarios & Goals

### Goal 1 - Data Preparation & Baseline (Priority: P1, Day 1)

Prepare the dataset and establish a TF-IDF baseline to compare against transformer-based approaches.

**Acceptance Criteria**:

1. **Given** the HuggingFace dataset, **When** download and subset script runs, **Then** `task_a_subset.parquet` exists with 20k stratified samples
2. **Given** the subset data, **When** TF-IDF + Logistic Regression trains, **Then** F1 score is printed (~0.60-0.70 expected)
3. **Given** raw code samples, **When** preprocessing function runs, **Then** comments are removed and whitespace is normalized
4. **Given** the preprocessing function, **When** tested on 100 samples, **Then** all samples transform without errors

---

### Goal 2 - CodeBERT Implementation (Priority: P1, Day 1)

Implement and train CodeBERT for Task A to establish a strong transformer baseline.

**Acceptance Criteria**:

1. **Given** CodeBERT tokenizer, **When** batch of 32 samples is tokenized, **Then** output shape is `(32, 512)`
2. **Given** `train_task_a.py`, **When** 1 epoch runs, **Then** script completes without crash
3. **Given** training for 3 epochs on 20k samples, **When** training completes, **Then** loss decreases and checkpoint is saved
4. **Given** trained model, **When** evaluated on validation set, **Then** F1 score is reported

---

### Goal 3 - Ablation & Analysis (Priority: P1, Day 2)

Conduct ablation study and generate analysis artifacts for the poster.

**Acceptance Criteria**:

1. **Given** CodeBERT model, **When** trained WITHOUT preprocessing (raw code), **Then** F1 with raw vs preprocessed code is compared
2. **Given** validation predictions, **When** per-language analysis runs, **Then** table with Python, Java, C++ F1 scores exists
3. **Given** predictions, **When** confusion matrix is generated, **Then** `confusion_matrix_task_a.png` exists
4. **Given** all baseline results, **When** bar chart is generated, **Then** `baseline_comparison.png` shows TF-IDF vs CodeBERT

---

### Goal 4 - Poster Creation (Priority: P1, Day 3)

Create an A0 academic poster with all findings and visualizations.

**Acceptance Criteria**:

1. **Given** LaTeX template, **When** compiled, **Then** `poster.pdf` is A0 format
2. **Given** poster content, **When** Introduction section is written, **Then** it contains ~150 words on problem statement
3. **Given** ablation results, **When** added to poster, **Then** table + 1 sentence insight is included
4. **Given** final poster, **When** exported, **Then** PDF is <20MB with both team members' names visible

---

## Requirements

### Functional Requirements

- **FR-001**: System MUST download SemEval-2026 Task 13 dataset from HuggingFace
- **FR-002**: System MUST create 20k stratified subset for Task A
- **FR-003**: System MUST implement TF-IDF + Logistic Regression baseline with max_features=10000, ngram_range=(1,3)
- **FR-004**: System MUST implement preprocessing function (comment removal, whitespace normalization)
- **FR-005**: System MUST use CodeBERT tokenizer with max_length=512
- **FR-006**: System MUST implement PyTorch DataLoader with batch_size=32
- **FR-007**: System MUST implement CodeBERT fine-tuning script with binary classification head
- **FR-008**: System MUST train for minimum 3 epochs on 20k samples
- **FR-009**: System MUST save model checkpoints to `outputs/models/model_task_a.pt`
- **FR-010**: System MUST generate confusion matrix visualization
- **FR-011**: System MUST compute per-language (Python, Java, C++) F1 scores
- **FR-012**: System MUST generate baseline comparison bar chart
- **FR-013**: System MUST create A0 LaTeX poster template
- **FR-014**: Poster MUST include: Title, Names, Introduction, Methodology, Results, Ablation, Analysis, Conclusion, References

### Technical Constraints

- Python 3.9+ with PyTorch and transformers
- CodeBERT model: `microsoft/codebert-base`
- Dataset: `DaniilOr/SemEval-2026-Task13`
- GPU training recommended (20k samples × 3 epochs × 512 tokens)
- Parquet format for data storage (efficient and column-oriented)

---

## Success Criteria

| Metric              | Target     | Rationale                                         |
| ------------------- | ---------- | ------------------------------------------------- |
| TF-IDF F1           | 0.60-0.70  | Lexical baseline (expected range)                 |
| CodeBERT F1         | 0.80+      | Semantic understanding should improve over TF-IDF |
| Improvement         | +15-20%    | Demonstrates transformer value                    |
| Ablation Δ          | Measurable | Preprocessing impact should be quantifiable       |
| Poster completeness | 100%       | All sections filled before deadline               |

---

## Key Outputs

### Data Artifacts

- `data/task_a_subset.parquet` - 20k stratified samples

### Model Artifacts

- `outputs/models/model_task_a.pt` - Trained CodeBERT checkpoint

### Visualization Artifacts

- `outputs/figures/confusion_matrix_task_a.png`
- `outputs/figures/loss_curve_a.png`
- `outputs/figures/baseline_comparison.png`

### Submission Artifacts

- `outputs/submissions/submission_task_a.csv` - Kaggle submission

### Documentation

- `poster/poster.tex` - LaTeX source
- `poster/poster.pdf` - Final A0 poster
- `notes/error_analysis.md` - Manual inspection notes

---

## Assumptions

- Dataset is publicly available on HuggingFace
- GPU access available (Google Colab Pro or university cluster)
- Zeev handles Task B independently (this spec covers only Anand's tasks)
- Default preprocessing is beneficial (hypothesis to test with ablation)
- CodeBERT pre-training transfers well to code classification task
