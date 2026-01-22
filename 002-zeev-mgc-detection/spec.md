# Feature Specification: Machine-Generated Code Detection (Zeev)

## Overview

This specification defines Zeev Tayer's scope for Team 32's SemEval-2026 Task 13 project (Machine-Generated Code Detection). The focus is Task B (multiclass per-generator classification) and the visualization and analysis responsibilities required for the poster.

Scope: Task B (11-class per-generator classification), per-generator analysis, Task B CodeBERT training pipeline, TF-IDF baseline for Task B, visualizations (confusion matrix, per-generator F1 table), and poster sections assigned to Zeev.

## Goals & Acceptance Criteria

- FR-Z01: Implement TF-IDF + Logistic Regression baseline for Task B and print Macro F1 on validation (expected baseline ~0.40-0.60 depending on class balance).
- FR-Z02: Implement `train_task_b.py` (CodeBERT fine-tuning for multiclass Task B), support `--preprocess` flag, run 1 epoch smoke test without crash.
- FR-Z03: Produce and save a per-generator F1 table (11 rows) as `outputs/002-zeev-mgc-detection/figures/per_generator_f1.csv` and a PNG/figure for the poster.
- FR-Z04: Produce and save a confusion matrix for Task B `outputs/002-zeev-mgc-detection/figures/confusion_matrix_task_b.png` (11×11) and store the matrix values as `outputs/002-zeev-mgc-detection/figures/confusion_matrix_task_b.npy`.
- FR-Z05: Identify top-3 confusion pairs with short explanations and save notes to `outputs/002-zeev-mgc-detection/notes/top3_confusions.md`.
- FR-Z06: Save all stdout prints, metric tables, and figure images under `/Users/benitarimbach/outputs/002-zeev-mgc-detection`.

## Key Outputs

- `spec.md`, `research.md`, `plan.md`, `tasks.md` (this folder)
- `src/train_task_b.py` (skeleton in repo `src/`)
- `outputs/002-zeev-mgc-detection/models/model_task_b.pt` (checkpoint placeholder)
- `outputs/002-zeev-mgc-detection/figures/confusion_matrix_task_b.png`
- `outputs/002-zeev-mgc-detection/figures/per_generator_f1.csv` and PNG
- `outputs/002-zeev-mgc-detection/submissions/submission_task_b.csv`

## Constraints & Notes

- Primary metric for Task B: Macro F1 (equal weight across 11 classes).
- Use raw `code` column for primary experiments; `--preprocess` option for ablation.
- Use `microsoft/codebert-base` as the pretrained encoder (same model family as CodeBERT). Use first-512 token truncation as default.
- Stratification rules: when sampling for Task B experiments, preserve label and language distribution where relevant; for per-generator metrics ensure enough samples per generator to compute meaningful F1 (if some classes are extremely small, note them in outputs).
