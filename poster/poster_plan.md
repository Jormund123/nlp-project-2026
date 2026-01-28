# Poster Implementation Plan: SemEval-2026 Task 13

## 1. Design Strategy

**Standard**: A0 Portrait (841mm x 1189mm)
**Tool**: LaTeX (`beamerposter` package)
**Theme**: Clean, 3-column layout (standard for German CS conferences).
**Visual Hierarchy**:

- **Title**: 72pt+ (Readable from 5m)
- **Headings**: 36pt+ (Sans-serif, Uni Bonn Blue)
- **Body**: 24pt+ (Serif/Sans-serif, high contrast)

## 2. Content Structure (The "Fingerprint Effect" Narrative)

### Header

- **Title**: Beyond the Surface: The Fingerprint Effect in AI Code Detection
- **Subtitle**: SemEval-2026 Task 13 (University of Bonn)
- **Authors**: Anand Karna, Zeev [Lastname]
- **Logos**: University of Bonn, SemEval

### Column 1: Motivation & Hypothesis

1. **The Problem**:
   - AI code is flooding repositories.
   - Detection is critical for academic integrity and security.
2. **The Hypothesis**:
   - AI models leave distinct "stylistic fingerprints" (formatting, comments).
   - _Hypothesis_: Removing these style features (preprocessing) will _degrade_ detection performance.
   - _Conventional Wisdom_: In standard NLP, preprocessing usually improves signal. Here, we argue style **is** the signal.

### Column 2: Methodology & Data

1. **Dataset Construction**:
   - **Source**: SemEval-2026 Task 13 (500k samples).
   - **Subset**: 20k Stratified Samples (Label + Language balanced).
   - **Split**: 80% Train, 20% Validation.
2. **Pipeline**:
   - **Baseline**: TF-IDF (N-gram 1-3) + LogReg.
   - **Model**: `microsoft/codebert-base` (First 512 tokens).
   - **Ablation Experiment**: Comparing **Raw Code** vs. **Preprocessed** (Comments/Whitespace removed).
3. **Training Config**:
   - Epochs: 3, Batch: 32, LR: 2e-5, Optimizer: AdamW.
   - Resource: Google Colab T4 GPU.

### Column 3: Results & Discussion

1. **Ablation Study (Testing the Hypothesis)**:
   - **Task A (Binary)**: F1 drops **1.3%** (0.985 → 0.972).
   - **Task B (Multi-class)**: F1 drops **3.6%** (0.721 → 0.686).
   - _Finding_: The "Fingerprint Effect" is stronger in multi-class problems; determining _which_ model generated code relies heavily on fine-grained style signals.
2. **Model Performance (Context)**:
   - **Task B**: CodeBERT (0.72) outperforms TF-IDF (0.45) by **+27%**.
   - _Visual_: Confusion Matrix (Human code is distinct; young models confuse with each other).
3. **Conclusion**:
   - **Key Takeaway**: Preprocessing removes discriminative artifacts. For AI detection, **style > semantics**.
   - **Limitation**: High performance (98%) likely reflects dataset-specific artifacts (GPT-family bias) rather than universal detection.

## 3. Technical Implementation Plan

### File Structure

```
poster/
├── main.tex              # Main LaTeX entry point
├── beamerthemeConf.sty   # Custom style file (blue/yellow accents)
├── figures/              # Symlinks to outputs/figures/
│   ├── loss_curve_a.png
│   ├── confusion_matrix_normalized.png
│   ├── baseline_comparison.png
│   └── per_generator_f1.png
└── Makefile              # Build automation
```

### LaTeX Packages

- `beamerposter`: Handles A0 scaling.
- `booktabs`: Professional tables.
- `tikz`: For flowcharts/diagrams.
- `pgfplots`: (Optional) for high-quality native plots if needed.

## 4. Verification Plan

1. **Compile Test**: Run `pdflatex` to ensure it builds.
2. **Visual Check**: Open PDF, zoom to 100%, verify legibility.
3. **Content Check**: Verify all numbers match `outputs/results/*.md`.
