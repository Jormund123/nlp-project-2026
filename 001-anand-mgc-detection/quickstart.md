# Developer Quickstart: MGC Detection (Anand)

**Feature**: Machine-Generated Code Detection  
**Last Updated**: 2026-01-22

This guide helps you quickly get started with the NLP project for detecting machine-generated code.

---

## Prerequisites

- Python 3.9+
- GPU with CUDA support (recommended for CodeBERT training)
- ~10GB disk space for dataset and models

---

## Quick Setup (10 minutes)

### 1. Create Virtual Environment

```bash
cd /path/to/nlp-project

# Create environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify GPU Access

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
# Should print: CUDA available: True
```

If no GPU, you can still run but training will be slow (~10x longer).

### 3. Download Dataset

```bash
python -c "
from datasets import load_dataset
ds = load_dataset('DaniilOr/SemEval-2026-Task13')
print(f'Dataset loaded: {len(ds[\"train\"])} samples')
"
```

### 4. Create Data Subset

```bash
python src/preprocess.py --create-subset --samples 20000
# Creates: data/task_a_subset.parquet
```

---

## Running the Baselines

### TF-IDF Baseline (5 min)

```bash
python src/baseline_tfidf.py

# Expected output:
# Training TF-IDF + LogReg baseline...
# Task A F1: 0.6X (your score will vary)
```

### CodeBERT Training (1-2 hours on GPU)

```bash
# Quick test (1 epoch, small sample)
python src/train_task_a.py --epochs 1 --samples 1000

# Full training
python src/train_task_a.py --epochs 3 --samples 20000

# Expected output:
# Epoch 1 Loss: X.XXX
# Epoch 2 Loss: X.XXX (should decrease)
# Epoch 3 Loss: X.XXX
# Saved checkpoint to outputs/models/model_task_a.pt
```

---

## Running Ablation Study

```bash
# With preprocessing (default)
python src/train_task_a.py --output outputs/models/model_preprocessed.pt

# Without preprocessing
python src/train_task_a.py --no-preprocess --output outputs/models/model_raw.pt

# Compare results
python src/evaluate.py --compare model_preprocessed.pt model_raw.pt
```

---

## Generating Visualizations

```bash
# Generate all figures
python src/evaluate.py --generate-figures

# Individual figures
python src/evaluate.py --confusion-matrix
python src/evaluate.py --loss-curve
python src/evaluate.py --baseline-comparison

# Output in: outputs/figures/
```

---

## Kaggle Submission

```bash
# Generate predictions
python src/evaluate.py --predict-test --output outputs/submissions/submission_task_a.csv

# Submit to Kaggle
# 1. Go to: https://www.kaggle.com/t/99673e23fe8546cf9a07a40f36f2cc7e
# 2. Upload submission_task_a.csv
# 3. Record your leaderboard score
```

---

## Poster Compilation

```bash
cd poster

# Compile LaTeX
pdflatex poster.tex
bibtex poster
pdflatex poster.tex
pdflatex poster.tex

# Check output
open poster.pdf  # On Mac
# xdg-open poster.pdf  # On Linux
```

---

## Common Issues & Fixes

### "CUDA out of memory"

Reduce batch size:

```bash
python src/train_task_a.py --batch-size 16  # or 8
```

### "ModuleNotFoundError: No module named 'transformers'"

Reinstall requirements:

```bash
pip install -r requirements.txt
```

### "Dataset not found"

Check HuggingFace connection:

```bash
python -c "from huggingface_hub import login; login()"
# Enter your HuggingFace token
```

### "Low F1 score (< 0.5)"

Check for:

- Class imbalance (is the data stratified?)
- Data leakage (are train/val properly separated?)
- Training issues (is loss decreasing?)

---

## Project Structure Reference

```
nlp-project/
├── data/
│   └── task_a_subset.parquet      # Your data goes here
├── src/
│   ├── preprocess.py              # Data loading
│   ├── baseline_tfidf.py          # TF-IDF baseline
│   ├── train_task_a.py            # CodeBERT training
│   └── evaluate.py                # Metrics & figures
├── outputs/
│   ├── models/                    # Saved checkpoints
│   ├── figures/                   # Generated PNGs
│   └── submissions/               # Kaggle CSVs
├── poster/
│   ├── poster.tex                 # LaTeX source
│   └── poster.pdf                 # Final output
└── notes/
    └── error_analysis.md          # Your analysis notes
```

---

## Verification Checklist

Run these to verify everything works:

```bash
# ✅ Environment
python --version                    # 3.9+
python -c "import torch; print(torch.__version__)"

# ✅ Data
ls data/task_a_subset.parquet

# ✅ TF-IDF baseline
python src/baseline_tfidf.py --test

# ✅ CodeBERT (quick test)
python src/train_task_a.py --epochs 1 --samples 100 --test-mode

# ✅ Figures
ls outputs/figures/*.png

# ✅ Poster
ls poster/poster.pdf
```

---

## Next Steps After Setup

1. ✅ Run TF-IDF baseline, record F1
2. ✅ Train CodeBERT, record F1
3. ✅ Run ablation, compare results
4. ✅ Generate all figures
5. ✅ Submit to Kaggle
6. ✅ Complete poster
7. ✅ Coordinate with Zeev for Task B integration

---

## Getting Help

- **README**: Check `README.md` for project overview
- **Spec**: See `002-anand-mgc-detection/spec.md` for requirements
- **Tasks**: See `002-anand-mgc-detection/tasks.md` for detailed checklist
- **Research**: See `002-anand-mgc-detection/research.md` for technical decisions

---

**Last Updated**: 2026-01-22  
**Maintained By**: Anand Karna
