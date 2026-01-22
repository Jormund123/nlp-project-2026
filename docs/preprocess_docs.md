# Preprocessing Design Decisions

**Date**: 2026-01-22  
**Module**: `src/preprocess.py`  
**Purpose**: Document the reasoning behind preprocessing implementation choices

---

## Summary

The preprocessing module uses a **conservative approach** that prioritizes data integrity over completeness. This document explains why.

---

## The "Fingerprint Paradox" Revisited

From `research.md`:

> "AI signatures live in comments, formatting, and whitespace patterns"

### Experiment Design

| Experiment   | Data Used                   | Purpose                                    |
| ------------ | --------------------------- | ------------------------------------------ |
| **Primary**  | Raw code (no preprocessing) | Preserve AI fingerprints                   |
| **Ablation** | Preprocessed code           | Test if removing "style" hurts performance |

---

## Why We DON'T Remove Multiline Comments/Docstrings

### The Problem with Regex

The `research.md` suggested:

```python
code = re.sub(r'""".*?"""', '', code, flags=re.DOTALL)  # Python docstrings
code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)  # C/Java block comments
```

**This breaks valid code:**

```python
# This is DATA, not a comment!
sql_query = """
    SELECT * FROM users
    WHERE active = 1
"""
# ^^^ Gets DELETED by the regex!
```

The regex cannot distinguish:

- `"""docstring"""` → comment-like, should remove
- `x = """multiline string"""` → data, must keep

### Options Considered

| Approach                        | Risk                    | Reward                    | Chosen? |
| ------------------------------- | ----------------------- | ------------------------- | ------- |
| Keep current (single-line only) | Miss some docstrings    | Safe, no data corruption  | ✅      |
| Add regex multiline removal     | **Destroys valid code** | Complete ablation         | ❌      |
| Use AST parser                  | Complex, Python-only    | Correct docstring removal | ❌      |

### Decision: Conservative Approach

We only remove **unambiguous single-line comments**:

- Python: `# comment` (not inside strings)
- C/Java/etc.: `// comment` (not inside strings)

**Rationale:**

1. **Speed doesn't matter** — preprocessing runs once
2. **Correctness matters more** — corrupted samples = bad training data
3. **Ablation still valid** — single-line comments capture most human informal markers (`# TODO`, `// HACK`, `# idk why this works`)

---

## What the Current Implementation Does

### `remove_comments(code, language)`

- Python: Removes `#` comments, handles string-awareness
- C-family: Removes `//` comments, basic string-awareness
- Unknown languages: Passes through unchanged

### `normalize_whitespace(code)`

- Tabs → 4 spaces
- Trailing whitespace removed
- Multiple blank lines collapsed to one
- Leading/trailing whitespace stripped

### `preprocess_code(code, language)`

- Combines both steps
- Used to generate `code_preprocessed` column

---

## Future Improvements (Post-Deadline)

If we wanted complete docstring removal for Python, we'd use the `ast` module:

```python
import ast

def remove_docstrings(code):
    """Remove docstrings using Python's AST parser."""
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            # Remove docstrings from functions/classes/modules
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                  ast.ClassDef, ast.Module)):
                if (node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)):
                    node.body.pop(0)
        return ast.unparse(tree)
    except SyntaxError:
        return code  # Return original if unparseable
```

**Why not now:**

- Only works for valid Python syntax
- Adds complexity
- Deadline: Jan 24, 2026

---

## Verification

Run preprocessing test:

```bash
python src/preprocess.py
```

Expected output:

- ✓ All 100 samples processed without errors
- Stats showing average length reduction
- Example before/after comparison
