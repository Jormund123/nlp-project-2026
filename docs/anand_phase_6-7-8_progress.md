# Phase 6-7-8 Decision Record: SemEval-2026 Task 13

**Author**: Anand Karna  
**Last Updated**: 2026-01-23  
**Purpose**: Document per-language analysis, visualization generation, and compute constraint decisions.

---

## 1. Phase 6: Per-Language Analysis

### 1.1 Results

| Language | Samples | F1 Score   | % of Dataset |
| -------- | ------- | ---------- | ------------ |
| Python   | 3,659   | **0.9778** | 91.5%        |
| C++      | 187     | 0.9239     | 4.7%         |
| Java     | 154     | 0.8970     | 3.8%         |

### 1.2 Hypothesis Outcome

**Original Hypothesis**: Python would be hardest to classify (most human variation in style/idioms).

**Actual Result**: Python is _easiest_ (highest F1), Java is hardest.

### 1.3 Why the Hypothesis Was Wrong

| Expected Reasoning              | Actual Cause                                        |
| ------------------------------- | --------------------------------------------------- |
| Python has most style variation | Dataset is 91% Python → model is best at Python     |
| Human Python has diverse idioms | Java/C++ underfitting due to 4% representation each |

**Key Insight**: The model's per-language performance reflects **data distribution**, not inherent language complexity. With only 154 Java samples, the model simply doesn't have enough examples to learn Java-specific patterns.

### 1.4 What This Reveals About the Dataset

The SemEval-2026 Task 13 dataset is heavily Python-dominated:

- This limits generalizability claims across languages
- Java/C++ results should be interpreted with caution (small sample)
- A balanced dataset would be needed to truly test cross-language AI detection

---

## 2. Phase 7: Visualization Generation

### 2.1 Figures Generated

| Figure              | File                          | Purpose                              |
| ------------------- | ----------------------------- | ------------------------------------ |
| Confusion Matrix    | `confusion_matrix_task_a.png` | Show TP/TN/FP/FN distribution        |
| Loss Curve          | `loss_curve_a.png`            | Compare raw vs preprocessed training |
| Baseline Comparison | `baseline_comparison.png`     | TF-IDF vs CodeBERT bar chart         |
| PR Curve            | `pr_curve_task_a.png`         | Precision-recall tradeoff            |

### 2.2 Visualization Standards Applied

```python
plt.style.use('seaborn-v0_8-whitegrid')
FIGURE_DPI = 300  # Poster-ready
FIGURE_SIZE = (8, 6)
COLORS = {
    'tfidf': '#1f77b4',      # Blue
    'codebert': '#ff7f0e',   # Orange
    'raw': '#2ca02c',        # Green
    'preprocessed': '#7f7f7f' # Gray
}
```

### 2.3 Key Visual Insights

**Loss Curve**: Raw code converges 40-54% faster than preprocessed across all epochs. This visually confirms the Fingerprint Paradox—stylistic features provide strong, early-learnable signals.

**Baseline Comparison**: The +18.6% gap between TF-IDF (0.8310) and CodeBERT (0.9854) demonstrates that semantic understanding of code provides significant gains beyond lexical patterns.

---

## 3. Phase 8: Compute Constraints & Sample Size Justification

### 3.1 The 20K vs 500K Decision

**Context**: The full SemEval-2026 Task 13 dataset contains ~500,000 samples. We used a 20,000 sample stratified subset.

**Decision**: Use 20K stratified sample.

**Evidence Level**: 1 (Informed heuristic based on compute constraints)

### 3.2 Free Colab T4 Limitations

| Resource   | Free Colab T4 | 500K Requirement  | Status          |
| ---------- | ------------- | ----------------- | --------------- |
| GPU RAM    | 15 GB         | ~15 GB (batch 32) | ⚠️ Borderline   |
| System RAM | 12.7 GB       | ~20-30 GB         | ❌ Insufficient |
| Runtime    | 12 hr max     | ~30-40 hours      | ❌ Impossible   |
| Disk       | ~100 GB       | ~5-10 GB          | ✅ OK           |

### 3.3 Time Estimates

```
20K samples × 3 epochs = ~2 hours ✅
500K samples × 3 epochs = ~50 hours ❌

Free Colab disconnects after 12 hours (often sooner)
→ Would reach ~40% through epoch 2 before timeout
```

### 3.4 What Happens If You Try 500K on Free Colab

1. **Loading phase**: RAM crash when loading 500K code strings (~20-30 GB)
2. **If RAM survives**: Training starts but session disconnects at ~40% of epoch 2
3. **Lost progress**: No checkpointing strategy can save you from session termination

### 3.5 Alternatives Considered

| Option             | Cost    | Feasibility              | Why Not Chosen            |
| ------------------ | ------- | ------------------------ | ------------------------- |
| Colab Pro          | $10/mo  | Likely works (25 GB RAM) | Budget constraint         |
| Colab Pro+         | $50/mo  | Definitely works         | Budget constraint         |
| University cluster | Free    | Best option              | Not available/configured  |
| Gradient streaming | Complex | Possible                 | Implementation time       |
| **20K subset**     | Free    | ✅ Chosen                | Time + budget constraints |

### 3.6 Justification for 20K Sample Size

1. **Stratified sampling preserves distribution**: Label AND language proportions maintained
2. **20K is statistically significant**: 16K train + 4K val provides stable F1 estimates
3. **Common practice in research**: Many papers use subsets due to compute constraints
4. **Focus on methodology, not scale**: The Fingerprint Paradox insight is valid regardless of scale

### 3.7 Limitation Statement for Poster

> **Compute Constraint**: Results based on 20K stratified sample (4% of full dataset) due to free-tier GPU limitations. Full dataset training would require Colab Pro (~25 GB RAM, extended sessions) or university compute cluster. Sample size is sufficient for methodology validation but may underestimate variance on underrepresented languages (Java: 3.8%, C++: 4.7%).

---

## 4. Summary of Decisions

| Phase   | Decision                       | Rationale                                          |
| ------- | ------------------------------ | -------------------------------------------------- |
| Phase 6 | Document per-language F1       | Reveals dataset imbalance, not language complexity |
| Phase 7 | Use 300 DPI, consistent colors | Poster-ready figures                               |
| Phase 8 | Use 20K sample                 | Free Colab cannot handle 500K (RAM + timeout)      |

---

## 5. Implications for Results Interpretation

### What We Can Claim

- ✅ "CodeBERT outperforms TF-IDF on this dataset"
- ✅ "Raw code preserves discriminative features"
- ✅ "Python detection is easier than Java/C++ on this sample"

### What We Cannot Claim

- ❌ "This model detects AI code with 98% accuracy in general"
- ❌ "Java is inherently harder to classify than Python"
- ❌ "These results scale to 500K samples"

---

## 6. Next Steps

| Task                  | Status   | Notes                                |
| --------------------- | -------- | ------------------------------------ |
| Kaggle submission     | Pending  | Generate predictions on test set     |
| Poster creation       | Pending  | Day 3 work                           |
| Full dataset training | Deferred | Requires Colab Pro or cluster access |

---

## Changelog

| Date       | Author | Change                                 |
| ---------- | ------ | -------------------------------------- |
| 2026-01-23 | Anand  | Initial version documenting phases 6-8 |
