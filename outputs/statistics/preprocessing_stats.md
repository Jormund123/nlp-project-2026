# Preprocessing Statistics

Generated: 2026-01-22 12:51:48

---

## Summary

| File | Samples | Original Avg | Preprocessed Avg | Reduction |
|------|---------|--------------|------------------|----------|
| task_a_train.parquet | 16000 | 821 chars | 789 chars | 3.8% |
| task_a_val.parquet | 4000 | 834 chars | 803 chars | 3.6% |
| task_a_subset.parquet | 20000 | 823 chars | 792 chars | 3.8% |

---

## Methodology

- **Comment removal**: Single-line only (`#` for Python, `//` for C-family)
- **Whitespace normalization**: Tabs → 4 spaces, trailing whitespace removed
- **Conservative approach**: No multiline/docstring removal to avoid data corruption

See [docs/preprocess_docs.md](../docs/preprocess_docs.md) for detailed rationale.
