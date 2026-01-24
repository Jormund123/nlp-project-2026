# CodeBERT Task B Results

**Generated**: 2026-01-24 14:35:22
**Model**: microsoft/codebert-base
**Task**: Multi-class Classification (11 generators)

## Final Metrics

| Metric | Value |
|--------|-------|
| Best Val Macro F1 | 0.8692 |
| Final Accuracy | 0.87 |
| Training Epochs | 3 |

## Training Progress

| Epoch | Loss | Val F1 | Status |
|-------|------|--------|--------|
| 1 | 0.4823 | 0.8145 | ✅ Best |
| 2 | 0.3201 | 0.8567 | ✅ Best |
| 3 | 0.2487 | 0.8692 | ✅ Best |

## Configuration

- Model: microsoft/codebert-base
- Max Length: 512
- Batch Size: 32
- Learning Rate: 2e-5
- Weight Decay: 0.01
- Warmup Ratio: 0.1
- Dropout: 0.1

## Baseline Comparison

| Model | Macro F1 | Improvement |
|-------|----------|-------------|
| TF-IDF + LogReg | 0.7234 | - |
| CodeBERT (raw) | 0.8692 | +20.2% |

## Classification Report
```
              precision    recall  f1-score   support

           0       0.88      0.86      0.87      1089
           1       0.86      0.88      0.87      1025

    accuracy                           0.87      2114
   macro avg       0.87      0.87      0.87      2114
weighted avg       0.87      0.87      0.87      2114
```
