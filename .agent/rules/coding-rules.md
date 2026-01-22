---
trigger: always_on
---

# NLP Project Workspace Rules

## Project Context

- **Project**: SemEval-2026 Task 13 - Machine-Generated Code Detection
- **Team**: Anand (Task A: Binary), Zeev (Task B: Multi-class)
- **Deadline**: January 24, 2026

## Key Decisions (DO NOT CHANGE)

- **Stratification**: By BOTH label AND language (not just label)
- **Primary experiment**: RAW code (no preprocessing)
- **Model**: microsoft/codebert-base with first 512 tokens
- **TF-IDF**: max_features=10000, ngram_range=(1,3), min_df=2, class_weight='balanced'
- **Training**: 3 epochs, lr=2e-5, warmup=10%, weight_decay=0.01, batch_size=32

## The "Fingerprint Paradox"

Raw code > preprocessed because AI fingerprints live in:

- Perfect docstrings, uniform 4-space indentation
- Standardized comments `"""This function computes..."""`
- Consistent variable naming (snake_case)

Human code has: tabs mixed with spaces, `# idk why this works`, typos

## Output Logging (CRITICAL)

ALL scripts MUST save outputs to markdown files:

- `outputs/statistics/` - Dataset stats, preprocessing stats
- `outputs/results/` - F1 scores, metrics, ablation results
- `outputs/logs/` - Training logs, epoch summaries

Format: Include timestamp, use markdown tables, reference methodology.

## Visualization Standards

```python
plt.style.use('seaborn-v0_8-whitegrid')
FIGURE_DPI = 300
FIGURE_SIZE = (8, 6)
FONT_SIZE = 12
COLORS = {'tfidf': '#1f77b4', 'codebert': '#ff7f0e', 'raw': '#2ca02c', 'preprocessed': '#7f7f7f'}
```
