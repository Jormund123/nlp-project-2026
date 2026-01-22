# CodeBERT Results - RAW Code (Task A)

**Generated**: 2026-01-22 16:29:44 (Colab)

## Metrics
| Metric | Value |
|--------|-------|
| **F1** | **0.9854** |
| ROC-AUC | 0.9988 |

## Configuration
- Seed: 42
- Model: microsoft/codebert-base
- Max length: 512
- Epochs: 3
- Batch size: 32
- Learning rate: 2e-05
- Dropout: 0.1

## Training Losses
- Epoch 1: 0.1819
- Epoch 2: 0.0532
- Epoch 3: 0.0326

## Classification Report
```
              precision    recall  f1-score   support

           0       0.98      0.98      0.98      1908
           1       0.99      0.99      0.99      2092

    accuracy                           0.98      4000
   macro avg       0.98      0.98      0.98      4000
weighted avg       0.98      0.98      0.98      4000

```
