# Phase 4-5 Decision Record: SemEval-2026 Task 13

**Author**: Anand Karna  
**Last Updated**: 2026-01-23  
**Purpose**: Document ablation study results, validate the "Fingerprint Paradox" hypothesis, and record technical decisions for Phases 4-5.

---

## 1. Ablation Study Design

### 1.1 The "Fingerprint Paradox" Hypothesis

**Context**: Traditional NLP preprocessing removes "noise" (formatting, comments, whitespace) to focus on semantic content. In machine-generated code detection, this approach may be counterproductive.

**Hypothesis**: AI-generated code contains stylistic "fingerprints" that exist in:

- **Comments**: Uniform `"""This function computes..."""` vs human `# idk why this works`
- **Formatting**: Perfect 4-space indentation vs inconsistent human style
- **Whitespace**: Standardized patterns vs messy human formatting

**Predicted Outcome**: Raw code will outperform preprocessed code.

### 1.2 Experimental Setup

| Configuration | Raw Code (Primary)        | Preprocessed (Ablation)                 |
| ------------- | ------------------------- | --------------------------------------- |
| Model         | `microsoft/codebert-base` | Same                                    |
| Input         | Original code             | Comments removed, whitespace normalized |
| Epochs        | 3                         | 3                                       |
| Batch Size    | 32                        | 32                                      |
| Learning Rate | 2e-5                      | 2e-5                                    |
| Hardware      | Google Colab T4 GPU       | Google Colab T4 GPU                     |

---

## 2. Results

### 2.1 Performance Comparison

| Metric              | Raw Code   | Preprocessed | Δ (Raw - Preprocessed) |
| ------------------- | ---------- | ------------ | ---------------------- |
| **F1 Score**        | **0.9854** | 0.9720       | **+1.34%** ✓           |
| ROC-AUC             | 0.9988     | 0.9953       | +0.35%                 |
| Precision (Class 0) | 0.98       | 0.96         | +2%                    |
| Precision (Class 1) | 0.99       | 0.98         | +1%                    |
| Recall (Class 0)    | 0.98       | 0.98         | 0%                     |
| Recall (Class 1)    | 0.99       | 0.96         | +3%                    |

**Verdict**: Raw code wins on all metrics. Hypothesis **confirmed**.

### 2.2 Training Dynamics

| Epoch | Raw Code Loss | Preprocessed Loss | Δ                        |
| ----- | ------------- | ----------------- | ------------------------ |
| 1     | 0.1819        | 0.3069            | Raw converges 40% faster |
| 2     | 0.0532        | 0.1112            | Raw 52% lower            |
| 3     | 0.0326        | 0.0705            | Raw 54% lower            |

**Observation**: Raw code not only achieves better final performance but also converges significantly faster. This suggests that stylistic features provide strong, learnable signals from the start.

---

## 3. Analysis: Why Does Preprocessing Hurt?

### 3.1 What Preprocessing Removes

The preprocessing pipeline performs:

1. **Comment Removal**: All single-line (`#`, `//`) and multi-line (`"""`, `/* */`) comments
2. **Whitespace Normalization**: Tabs → 4 spaces, trailing whitespace removed, multiple blank lines collapsed

### 3.2 Evidence: What We're Losing

**AI-Generated Patterns** (destroyed by preprocessing):

- Perfect docstring formatting: `"""Calculate the sum of two numbers."""`
- Consistent 4-space indentation throughout
- Standardized comment style: `# Initialize variables`
- No trailing whitespace, exactly 2 blank lines between functions

**Human Patterns** (also destroyed):

- Inconsistent indentation: mix of tabs and spaces
- Informal comments: `# this shouldn't work but it does`
- Typos in variable names and comments
- Extra blank lines, trailing spaces

By removing these patterns, we force the model to rely only on **semantic structure**, which is more similar between human and AI code than stylistic patterns.

### 3.3 Theoretical Explanation

| Feature Type | Human Code       | AI Code          | Discriminative Power |
| ------------ | ---------------- | ---------------- | -------------------- |
| **Logic**    | Variable         | Standardized     | Medium               |
| **Style**    | Inconsistent     | Perfect          | **High**             |
| **Comments** | Informal, sparse | Formal, complete | **High**             |

> **Key Insight**: The "logic" (algorithms, data structures) is often the most human-like part of AI code—LLMs are trained on human code. The discriminative signal comes from **how** the code is formatted and documented, not **what** it does.

---

## 4. Decision Record

### 4.1 Decision: Use Raw Code for Final Model

**Evidence Level**: 4 (Empirical comparison on our data)

**Rationale**:

- +1.34% F1 improvement is meaningful at this performance level
- Faster convergence reduces training cost
- Simpler pipeline (no preprocessing step)

**Tradeoffs Accepted**:

- Model may be more sensitive to formatting changes
- Harder to explain to non-technical stakeholders

**Reversal Trigger**: If test set shows preprocessing performs better (indicates overfitting to training set formatting patterns).

### 4.2 Alternative Not Taken: Hybrid Approach

We considered training on both raw and preprocessed versions and ensembling.

**Why Rejected**:

- 2x training cost
- Marginal expected improvement (raw already at 98.5% F1)
- Time constraint (deadline: Jan 24)

---

## 5. Implications for Poster

### 5.1 Key Talking Point

> "We confirm the **Fingerprint Paradox**: AI-generated code is most detectable not by what it does, but by how it's formatted. Raw code outperforms preprocessed by 1.34% F1 because AI leaves stylistic fingerprints in comments, indentation, and whitespace."

### 5.2 Visualization Recommendation

A side-by-side comparison showing:

- Human code: `# todo: fix edge case`, tabs and spaces mixed
- AI code: `"""Handles the edge case for invalid inputs."""`, perfect 4-space indent

---

## 6. Comparison with Baseline

### 6.1 Full Model Comparison

| Model                   | Task A F1  | ROC-AUC    | Training Time |
| ----------------------- | ---------- | ---------- | ------------- |
| TF-IDF + LogReg         | 0.8310     | 0.9107     | ~30 seconds   |
| CodeBERT (preprocessed) | 0.9720     | 0.9953     | ~2 hours      |
| **CodeBERT (raw)**      | **0.9854** | **0.9988** | ~2 hours      |

### 6.2 Analysis

- **TF-IDF → CodeBERT (raw)**: +18.58% F1 improvement
- **Preprocessed → Raw**: +1.34% F1 improvement (confirms hypothesis)
- **Semantic features matter**: The jump from TF-IDF to CodeBERT shows that contextual understanding of code provides significant gains beyond lexical patterns.

---

## 7. Training Log Summary

### 7.1 Raw Code Training (Primary)

```
Generated: 2026-01-22 16:29:44 (Colab T4 GPU)
Configuration:
- Seed: 42
- Model: microsoft/codebert-base
- Max length: 512
- Epochs: 3
- Batch size: 32
- Learning rate: 2e-05
- Dropout: 0.1

Training Losses:
- Epoch 1: 0.1819
- Epoch 2: 0.0532
- Epoch 3: 0.0326

Final Metrics:
- F1: 0.9854
- ROC-AUC: 0.9988
```

### 7.2 Preprocessed Code Training (Ablation)

```
Generated: 2026-01-23 16:38:38 (Colab T4 GPU)
Configuration: Same as above

Training Losses:
- Epoch 1: 0.3069
- Epoch 2: 0.1112
- Epoch 3: 0.0705

Final Metrics:
- F1: 0.9720
- ROC-AUC: 0.9953
```

---

## 8. Next Steps

| Task                    | Status  | Priority |
| ----------------------- | ------- | -------- |
| Per-language analysis   | Pending | High     |
| Generate visualizations | Pending | High     |
| Kaggle submission       | Pending | Medium   |
| Poster creation         | Pending | Critical |

---

## 9. Critical Limitations: Why 98.5% F1 is Misleading

> [!CAUTION]
> A 98.5% F1 score does **not** mean the model can detect all AI-generated code. This section documents why.

### 9.1 The Core Problem: Dataset-Specific Patterns

The model achieves high accuracy because it learns **artifacts of this specific dataset**, not generalizable AI detection:

| What the Model Learns                     | What We Claim It Learns        |
| ----------------------------------------- | ------------------------------ |
| Patterns from GPT-3.5/GPT-4 specifically  | "AI-generated code" in general |
| Artifacts of the prompting templates used | Universal AI fingerprints      |
| Dataset curation decisions                | Real-world code distribution   |

### 9.2 Why Same-Distribution Validation is Deceptive

Our validation set comes from the **same distribution** as training:

- Same LLMs (likely GPT family)
- Same prompting style ("Write a Python function that...")
- Same time period (pre-dataset-cutoff)

**What this measures**: Can we detect code from the LLMs in our training data?  
**What it doesn't measure**: Can we detect code from:

- Future LLM versions (GPT-5, Claude-4, etc.)
- Different prompting styles
- Humans who write in "clean" AI-like style
- AI code that was manually edited

### 9.3 The Fingerprint Paradox Limitation

The "Fingerprint Paradox" is real—raw code outperforms preprocessed by 1.34%. However:

| Claim                             | Reality                                              |
| --------------------------------- | ---------------------------------------------------- |
| "AI has stylistic fingerprints"   | True, but fingerprints are **model-specific**        |
| "Preprocessing destroys signals"  | True for **this dataset's AI patterns**              |
| "This generalizes to all AI code" | **False**—different LLMs have different fingerprints |

### 9.4 Expected Real-World Performance

| Scenario                                 | Expected F1           |
| ---------------------------------------- | --------------------- |
| Same LLMs, same prompts (our validation) | ~98%                  |
| Same LLMs, different prompts             | ~85-90% (estimated)   |
| Different LLMs (e.g., Claude, Gemini)    | ~60-75% (estimated)   |
| Adversarial AI code (human-edited)       | ~50-60% (near random) |

### 9.5 Academic Honesty Statement

For the poster and any academic submission, we must acknowledge:

> **Limitation**: Our model achieves 98.5% F1 on the SemEval-2026 Task 13 validation set. This measures detection of code from specific LLMs (likely GPT-family) under specific prompting conditions. Performance on unseen LLMs or adversarially-crafted code is expected to be significantly lower. The "Fingerprint Paradox" finding (raw > preprocessed) is valid but dataset-specific.

### 9.6 What We Can Defensibly Claim

| ✅ Valid Claims                              | ❌ Invalid Claims                      |
| -------------------------------------------- | -------------------------------------- |
| "98.5% F1 on SemEval Task 13"                | "Can detect AI code with 98% accuracy" |
| "Raw code outperforms preprocessed by 1.34%" | "Raw code always beats preprocessing"  |
| "CodeBERT outperforms TF-IDF by 18.5%"       | "Semantic models always beat lexical"  |
| "AI code has detectable patterns"            | "All AI code is detectable"            |

---

## 10. References

- Feng, Z., et al. (2020). CodeBERT: A Pre-Trained Model for Programming and Natural Languages. [arXiv:2002.08155](https://arxiv.org/abs/2002.08155)
- SemEval-2026 Task 13 Dataset: [DaniilOr/SemEval-2026-Task13](https://huggingface.co/datasets/DaniilOr/SemEval-2026-Task13)

---

## Changelog

| Date       | Author | Change                                             |
| ---------- | ------ | -------------------------------------------------- |
| 2026-01-23 | Anand  | Initial version documenting ablation study results |
