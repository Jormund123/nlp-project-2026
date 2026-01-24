# TF-IDF Baseline Results (Task A)

**Generated**: 2026-01-22 13:51:17

## Metrics
| Metric | Value |
|--------|-------|
| **F1** | **0.8310** |
| ROC-AUC | 0.9107 |

## Configuration
- `max_features`: 10000
- `ngram_range`: (1, 3)
- `min_df`: 2
- `model`: LogisticRegression(balanced)

## Detailed Report
```
              precision    recall  f1-score   support

           0       0.78      0.91      0.84      1908
           1       0.90      0.77      0.83      2092

    accuracy                           0.84      4000
   macro avg       0.84      0.84      0.84      4000
weighted avg       0.85      0.84      0.84      4000

```
