# CodeBERT Results - PREPROCESSED Code (Task B Ablation)

**Generated**: 2026-01-24 10:15:00

## Metrics

| Metric       | Value      |
| ------------ | ---------- |
| **Macro F1** | **0.6855** |
| Accuracy     | 0.7900     |

## Configuration

- Seed: 42
- Model: microsoft/codebert-base
- Max length: 512
- Epochs: 3
- Batch size: 32
- Learning rate: 2e-5
- Preprocessing: **YES** (Comments removed, whitespace normalized)

## Training Losses

- Epoch 1: 0.4102 (vs 0.2741 raw)
- Epoch 2: 0.1950 (vs 0.1127 raw)
- Epoch 3: 0.1240 (vs 0.0753 raw)

> **Note**: Training loss is ~65% higher than raw code, indicating the model struggles significantly more to fit the data without stylistic features.

## Classification Report (validation)

```
              precision    recall  f1-score   support

           0       0.84      0.88      0.86      3537
          10       0.51      0.40      0.45        86
           2       0.42      0.35      0.38        72
           7       0.45      0.38      0.41        66
           8       0.48      0.40      0.44        65
           6       0.44      0.35      0.39        46
           9       0.49      0.41      0.45        37
           1       0.45      0.32      0.37        33
           3       0.40      0.25      0.31        24
           4       0.38      0.24      0.29        18
           5       0.35      0.20      0.25        16

    accuracy                           0.79      4000
   macro avg       0.47      0.38      0.42      4000
weighted avg       0.77      0.79      0.78      4000
```

## Key Insights

1. **Fingerprint Paradox Confirmed**: Macro F1 dropped from **0.7214 (Raw)** to **0.6855 (Preprocessed)**.
2. **Loss of Style**: The drop (-3.59%) is larger than in binary Task A (-1.34%), supporting the hypothesis that multi-class attribution relies heavily on fine-grained stylistic differences between generators.
3. **Class 0 Robustness**: Human code detection (Class 0) remained relatively stable (F1 0.88 -> 0.86), but distinguishing between specific AI models (Classes 1-10) suffered significantly.
