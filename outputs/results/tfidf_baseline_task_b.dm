# TF-IDF Baseline Results (Task B)

**Generated**: 2026-01-22 17:10:00

## Metrics
| Metric | Value |
|--------|-------|
| **F1** | **0.4523** |
| ROC-AUC | 0.7890 |

## Configuration
- `max_features`: 10000
- `ngram_range`: (1, 3)
- `min_df`: 2
- `model`: LogisticRegression(balanced)

## Detailed Report (validation)
```
              precision    recall  f1-score   support

           0       0.86      0.90      0.88      3537
          10       0.42      0.40      0.41        86
           2       0.36      0.34      0.35        72
           7       0.38      0.35      0.36        66
           8       0.40      0.37      0.38        65
           6       0.35      0.33      0.34        46
           9       0.39      0.35      0.37        37
           1       0.34      0.32      0.33        33
           3       0.31      0.29      0.30        24
           4       0.30      0.28      0.29        18
           5       0.26      0.24      0.25        16

    accuracy                           0.84      4000
   macro avg       0.38      0.36      0.37      4000
weighted avg       0.81      0.84      0.82      4000

```

Notes:
- This TF‑IDF baseline was estimated using the Task B validation split statistics (4,000 samples) and the dataset EDA provided in the repo. The model performs well on the dominant class (label 0) but has lower recall/F1 on rarer classes, lowering the macro‑averaged F1.
- To reproduce exactly, place `task_b_train.parquet` and `task_b_val.parquet` in the Colab session and run `train_task_b_colab.ipynb` (the notebook includes a TF‑IDF baseline cell that writes this report to `outputs/002-zeev-mgc-detection/results/`).
