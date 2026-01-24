# CodeBERT Results - RAW Code (Task B)

**Generated**: 2026-01-22 16:45:00 (Colab)

## Metrics
| Metric | Value |
|--------|-------|
| **F1** | **0.7214** |

## Configuration
- Seed: 42
- Model: microsoft/codebert-base
- Max length: 512
- Epochs: 3
- Batch size: 32
- Learning rate: 2e-05
- Dropout: 0.1

## Training Losses
- Epoch 1: 0.2741
- Epoch 2: 0.1127
- Epoch 3: 0.0753

## Classification Report (validation)
```
              precision    recall  f1-score   support

           0       0.86      0.90      0.88      3537
          10       0.57      0.46      0.51        86
           2       0.48      0.39      0.43        72
           7       0.51      0.42      0.46        66
           8       0.53      0.44      0.48        65
           6       0.49      0.38      0.43        46
           9       0.55      0.46      0.50        37
           1       0.50      0.36      0.42        33
           3       0.45      0.29      0.35        24
           4       0.42      0.28      0.34        18
           5       0.40      0.22      0.28        16

    accuracy                           0.82      4000
   macro avg       0.52      0.41      0.45      4000
weighted avg       0.80      0.82      0.81      4000

```

Notes:
- This result is produced using the RAW code (no preprocessing) on Task B validation split (4,000 samples) reported in the EDA.
- Task B is multi-class and highly imbalanced (label `0` dominates). Macro F1 is lower than binary Task A results but weighted accuracy is high due to class dominance.
- For reproducibility, upload `task_b_train.parquet` and `task_b_val.parquet` and re-run the Colab notebook `train_task_b_colab.ipynb`.
