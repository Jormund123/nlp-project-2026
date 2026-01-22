# EDA Results — task_a_subset.parquet and task_b_subset.parquet

Generated: 2026-01-21
Source: https://github.com/Jormund123/nlp-project-2026/tree/master/data

This file contains the printed exploratory data analysis (EDA) statistics produced by `/Users/benitarimbach/eda_task_parquet.py`.

---

## task_a_subset.parquet

- Rows: 20000
- Columns: ['code', 'generator', 'label', 'language']
- Text column used: `code`
  - Avg length (chars): 870.46
  - Median length: 462.50
  - Min length: 1
  - Max length: 475006

- Label counts:
  - 1: 10461
  - 0: 9539

- Language counts:
  - Python: 18270
  - C++: 975
  - Java: 755

---

## task_b_subset.parquet

- Rows: 20000
- Columns: ['code', 'generator', 'label', 'language']
- Text column used: `code`
  - Avg length (chars): 1417.47
  - Median length: 812.00
  - Min length: 40
  - Max length: 15354

- Label counts:
  - 0: 17684
  - 10: 432
  - 2: 360
  - 7: 328
  - 8: 325
  - 6: 231
  - 9: 184
  - 1: 167
  - 3: 121
  - 4: 89
  - 5: 79

- Language counts:
  - Python: 5492
  - Java: 5376
  - C#: 2579
  - JavaScript: 1647
  - C++: 1509
  - Go: 1170
  - PHP: 1167
  - C: 1060

---

Notes:
- The script autodetected `code` as the text column and `label`/`language` as metadata columns.
- `task_a` contains a very large max code length (475,006 chars) — consider inspecting or capping outliers.
- `task_b` labels are multi-class and highly imbalanced (label `0` dominates).

