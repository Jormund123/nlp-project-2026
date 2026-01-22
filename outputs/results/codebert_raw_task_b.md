# TF-IDF Baseline Results (Task B)

**Generated**: 2026-01-22 14:00:00

## Metrics
| Metric | Value |
|--------|-------|
| **F1** | **0.0000** |

## Configuration
- `max_features`: 10000
- `ngram_range`: (1, 3)
- `min_df`: 1
- `model`: LogisticRegression(balanced)

## Detailed Report
```
              precision    recall  f1-score   support

           0       0.00      0.00      0.00       0.0
           1       0.00      0.00      0.00       1.0
           2       0.00      0.00      0.00       1.0

    accuracy                           0.00       2.0
   macro avg       0.00      0.00      0.00       2.0
weighted avg       0.00      0.00      0.00       2.0

```

Notes:
- This run used the synthetic demo (no dataset present at `data/task_b_subset.parquet`).
- Expect different metrics when running on a real Task B subset (provide `data/task_b_subset.parquet` or permit dataset creation).
