# NLP Project Audit Rules

## 🔍 Data Integrity Audit

- [ ] **Leakage Check**: Are any train samples in validation/test?
- [ ] **Stratification Check**: Does `val` set language distribution match `train` within 2%?
- [ ] **Sanity Check**: Are `code` columns strings and not objects/bytes?
- [ ] **Preprocessing**: Is `code_preprocessed` actually different from `code`? (Run diff check)

## 🧪 Experiment Audit

- [ ] **Baseline Check**: Is TF-IDF F1 > 0.50? (If not, something is broken)
- [ ] **Comparison**: Is CodeBERT > TF-IDF? (Hypothesis validation)
- [ ] **Ablation**: Is Raw > Preprocessed? (Fingerprint Paradox validation)
- [ ] **Reproducibility**: Is `random_state=42` used everywhere?

## 📂 Output & Artifact Audit

- [ ] **No Silent Failures**: process exiting with 0 but printing "Error"?
- [ ] **Traceability**: Do output files include the timestamp?
- [ ] **Visuals**: Do all plots use the project style (`seaborn-v0_8-whitegrid`)?
- [ ] **Markdown**: Are stats saved to `outputs/statistics/*.md`, not just printed?

## 📝 Spec Compliance

- [ ] **Naming**: Do filenames match `plan.md`? (e.g., `model_task_a_raw.pt`)
- [ ] **Config**: matches `rules` (e.g., `lr=2e-5`)?

Use this checklist before marking a task as COMPLETED.
