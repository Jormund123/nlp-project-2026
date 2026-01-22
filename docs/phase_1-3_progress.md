# Phase 1-3 Decision Record: SemEval-2026 Task 13

**Author**: Anand Karna  
**Last Updated**: 2026-01-22  
**Purpose**: A defensible record of technical decisions, rationale, and tradeoffs for the Machine-Generated Code Detection system (Task A).

---

## 1. Foundation & Metric Selection

### 1.1 Choice of Primary Metric: F1 Score

**Context**: We are building a binary classifier (Human vs. machine). The dataset is balanced (50/50), but real-world scenarios are often imbalanced.

**Options Considered**:

1. **Accuracy**: Simple, interpretable.
   - _Failure Mode_: If 90% of code is human, a "always predict human" model gets 90% accuracy but 0 utility.
2. **ROC-AUC**: Good for threshold-independent evaluation.
   - _Status_: Used as secondary metric.
3. **F1 Score** (Selected): Harmonic mean of Precision and Recall.
   - _Why_: In academic integrity settings, **False Positives** (accusing a student falsely) and **False Negatives** (missing AI code) are both costly. F1 forces a balance.

**Decision**: Optimize for **F1 Score**.

---

## 2. Approach 1: Lexical Baseline (TF-IDF)

### 2.1 Why TF-IDF?

**Context**: Before training expensive transformers, we need a baseline to quantify "how hard is this task?" and "does semantics matter?".

**Decision**: Use **TF-IDF (Term Frequency-Inverse Document Frequency) + Logistic Regression**.

**Rationale (Level 2 - Theoretical)**:

- **Hypothesis**: AI models tend to over-use specific control structures (`try/except` blocks, standard variable names). TF-IDF captures these keywords.
- **Efficiency**: Trains in <1 minute vs hours for CodeBERT.

### 2.2 N-gram Range Selection: (1, 3)

**Options**:

1. **Unigrams (1,1)**: Just words (`def`, `return`).
   - _Cons_: Loses context (e.g., cannot distinguish `x = 1` vs `x == 1`).
2. **Bigrams (1,2)**: `if x`, `for i`.
3. **Trigrams (1,3)** (Selected): `for i in`, `if name ==`.
   - _Pros_: Captures short syntactic idioms common in code.

**Outcome**:

- **F1 = 0.8310**: Validates that lexical features are VERY predictive. This sets a high bar for CodeBERT.

---

## 3. Approach 2: Semantic Classifier (CodeBERT)

### 3.1 Why CodeBERT?

**Context**: Standard BERT is trained on English text (Wikipedia). Code has specific structure (keywords, indentation, syntax).

**Options Considered**:

1. **BERT-base**: Trained on English.
   - _Cons_: Tokenizer breaks code into nonsense subwords.
2. **GraphCodeBERT**: Includes Data Flow Graph (DFG).
   - _Cons_: Requires complex AST parsing (language-specific). High implementation risk.
3. **CodeBERT** (Selected): Pre-trained on CodeSearchNet (6 languages).
   - _Rationale (Level 3 - Literature)_: Standard SOTA baseline for code tasks (Feng et al., 2020). Transfers well to classification.

### 3.2 Input Length Constraint (512 vs 1024)

**Constraint**: `microsoft/codebert-base` has a max position embedding of 512 tokens.

**Options**:

1. **Truncate Head** (First 512 tokens) (Selected).
   - _Rationale_: "Fingerprint Paradox" — imports, docstrings, and function signatures (top of file) contain the most stylistic signals.
2. **Sliding Window**: Process chunks and average specific vectors.
   - _Tradeoff_: Implementation complexity vs marginal gain. Deferred to future work.

**Decision**: **First 512 tokens**.

---

## 4. The "Fingerprint Paradox" (Preprocessing Strategy)

### 4.1 The Problem

In standard NLP, we clean text (lowercase, remove punctuation) to focus on meaning. In AI detection, **formatting IS the signal**.

**Hypothesis**:

- **Humans**: Messy indentation, typos, customized comments (`# todo fix this`).
- **AI**: Perfect 4-space indent, capitalized comments, standard variable names.

### 4.2 Decision: Raw Code

**Decision**: Train primarily on **RAW code**.

- **Evidence**: Start with Raw.
- **Ablation**: Compare with "Cleaned" version (comments removed).
- **Expectation**: Raw F1 > Cleaned F1.

---

## 5. Technical Implementation Details

### 5.1 Variable Hidden Size

**Why**: In `CodeBERTClassifier`, we use `auto_model.config.hidden_size` instead of `768`.

- If we swap CodeBERT for **DistilRoBERTa** (hidden=768) or **CodeBERT-Large** (hidden=1024), the code breaks if hardcoded.
- _Principle_: Loose coupling.

### 5.2 Reproducibility Stack

**Problem**: Transformers are non-deterministic on GPU by default.
**Solution**:

```python
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
g = torch.Generator(); g.manual_seed(42)  # For DataLoader
```

**Cost**: Slower training (deterministic algorithms disable some CUDA optimizations).
**Benefit**: We can debug regressions confidently.

---

## 6. Current Status & Next Steps

| Component    | Status     | Result/Note                    |
| ------------ | ---------- | ------------------------------ |
| **Data**     | ✅ Done    | 20k stratified samples         |
| **TF-IDF**   | ✅ Done    | **F1: 0.83** (Strong Baseline) |
| **CodeBERT** | 🔄 Running | Training on MPS (Epoch 1/3)    |

**Immediate Next Step**:

- Evaluate CodeBERT.
- If F1 < 0.83, investigate "Catastrophic Forgetting" or Hyperparameters.
- If F1 > 0.83, proceed to Error Analysis.
