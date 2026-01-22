---
description: Follow these guidelines whenever writing a documentation after an implementation.
---

# Documentation Writing Rules

> **Purpose**: A thinking framework for writing defensible technical documentation.  
> **Philosophy**: Teach the mindset, not just the format. Good documentation comes from asking the right questions.

---

## 1. The Core Mindset

### 1.1 The Defensibility Principle

Documentation exists to answer one meta-question:

> _"If someone knowledgeable challenges any decision, can you explain your reasoning with evidence and awareness of tradeoffs?"_

This applies whether the challenger is a professor, a code reviewer, your future self, or a teammate.

### 1.2 The Decision Spectrum

Not everything needs the same depth. Calibrate effort to impact:

```
Low Impact                                              High Impact
    │                                                        │
    ▼                                                        ▼
Variable names ──── Utility functions ──── Architecture ──── Core algorithms
    │                                                        │
    └── Minimal docs                       Extensive docs ───┘
```

**Rule of thumb**: If changing it would require >1 hour of work or affect results by >1%, document the reasoning.

---

## 2. The Question-Generation Framework

Good documentation answers questions before they're asked. Use these patterns to generate the right questions for ANY technical decision.

### 2.1 The Five Whys of Any Decision

For any choice X, ask:

| #   | Question Pattern              | What It Reveals          |
| --- | ----------------------------- | ------------------------ |
| 1   | **What** is X?                | Clarity of understanding |
| 2   | **Why** X over alternatives?  | Decision rationale       |
| 3   | **What** alternatives exist?  | Awareness of options     |
| 4   | **What** evidence supports X? | Rigor of reasoning       |
| 5   | **When** would X be wrong?    | Intellectual honesty     |

### 2.2 Question Templates by Category

Use these templates to generate project-specific questions:

**For any parameter/configuration:**

- Why this value instead of [higher/lower]?
- What happens if this changes by 10x? By 0.1x?
- Is this value derived, tuned, or assumed?
- What resource constraints influenced this?

**For any architectural choice:**

- What problem does this solve that simpler approaches don't?
- What are we trading away by using this?
- How does this scale with [data size / complexity / users]?
- What would we use if we had [more time / more compute / different constraints]?

**For any data processing step:**

- What information is preserved? What is lost?
- Is this reversible? Does it need to be?
- How does this affect downstream components?
- What assumptions does this encode?

**For any tool/library choice:**

- Why this over the obvious alternative?
- What's the maintenance/compatibility risk?
- Could we swap this out later if needed?

### 2.3 The Constraint-Aware Questions

Many decisions are forced by constraints. Document both the decision AND the constraint:

```
DECISION: We did X
    │
    ├── Because we CHOSE to (preference) ──→ Explain reasoning
    │
    └── Because we HAD to (constraint) ──→ Document the constraint
         │
         ├── Resource constraint (time, memory, budget)
         ├── Dependency constraint (library limitations)
         ├── Requirement constraint (specs, compatibility)
         └── Knowledge constraint (learning curve, expertise)
```

---

## 3. Levels of Justification

Rank evidence by strength. Match evidence strength to decision importance.

| Level | Evidence Type                        | Example                                            | Use For                                  |
| ----- | ------------------------------------ | -------------------------------------------------- | ---------------------------------------- |
| **5** | Empirical + Statistical significance | "A/B test: p<0.05, effect=+2.3%"                   | Critical decisions                       |
| **4** | Empirical comparison                 | "X gave 0.82 vs Y's 0.79 on our data"              | Important decisions                      |
| **3** | Literature/external validation       | "Recommended in [Paper], validated in [Benchmark]" | Standard choices                         |
| **2** | Theoretical reasoning                | "Smaller X leads to Y because [mechanism]"         | Reasonable defaults                      |
| **1** | Informed heuristic                   | "Common practice in this domain"                   | Minor decisions                          |
| **0** | No justification                     | "We used X"                                        | ❌ Unacceptable for anything non-trivial |

**Guideline**: Critical decisions need Level 4-5. Standard decisions need Level 2-3. Only trivial decisions can be Level 1.

---

## 4. Documentation Patterns

### 4.1 The Minimal Decision Record

For routine decisions, a lightweight format:

```markdown
**[Topic]**: [Choice]  
**Why**: [One-sentence rationale]  
**Alternative**: [What you'd do differently under different constraints]
```

### 4.2 The Full Decision Record

For significant decisions, expand to:

```markdown
### [Decision Topic]

**Context**: [What situation required this decision]

**Options Considered**:

- Option A: [Pros] / [Cons]
- Option B: [Pros] / [Cons]

**Decision**: [What was chosen]

**Rationale**: [Why—with evidence level noted]

**Tradeoffs Accepted**: [What you gave up]

**Reversal Trigger**: [What would make you reconsider]
```

### 4.3 The Comparison Table

When choosing between alternatives, tables force clear thinking:

```markdown
| Criterion   | Option A | Option B | Winner   |
| ----------- | -------- | -------- | -------- |
| [Metric 1]  | ...      | ...      | ...      |
| [Metric 2]  | ...      | ...      | ...      |
| **Overall** |          |          | [Choice] |
```

---

## 5. Writing Quality Principles

### 5.1 Precision Over Vagueness

| ❌ Vague               | ✅ Precise               |
| ---------------------- | ------------------------ |
| "significantly better" | "4.2% improvement"       |
| "large dataset"        | "142K samples"           |
| "fast"                 | "23 minutes"             |
| "we tried several"     | "we evaluated {A, B, C}" |
| "standard approach"    | "following [source]"     |

### 5.2 Acknowledge Uncertainty

Use calibrated language:

| Confidence | Language Examples                         |
| ---------- | ----------------------------------------- |
| High       | "X causes Y", "evidence shows"            |
| Medium     | "X likely contributes", "results suggest" |
| Low        | "X may influence", "we hypothesize"       |

### 5.3 Document Failures Too

Survivorship bias is a documentation anti-pattern. Include:

- Approaches that didn't work (and why you think they failed)
- Hypotheses that were disproven
- Dead ends that others shouldn't repeat

---

## 6. Red Flags to Catch

### 6.1 Phrases That Need Expansion

When you write these, stop and expand:

| Red Flag            | Question to Ask Yourself                       |
| ------------------- | ---------------------------------------------- |
| "We just used..."   | Why that specific choice?                      |
| "Obviously..."      | Is it obvious? Would a newcomer know?          |
| "Standard practice" | Standard where? Why is it standard?            |
| "It didn't work"    | How didn't it work? What metrics?              |
| "Better results"    | How much better? By what measure?              |
| "We had to..."      | What forced this? Can you name the constraint? |

### 6.2 Reasoning Anti-Patterns

| Anti-Pattern         | Problem                          | Fix                              |
| -------------------- | -------------------------------- | -------------------------------- |
| Circular reasoning   | "X is good because X works well" | Explain the mechanism            |
| Appeal to authority  | "Google does this"               | Explain why it fits YOUR context |
| Missing alternatives | Only one option discussed        | Show you considered others       |
| Hindsight bias       | "Obviously we needed..."         | Document what you actually tried |
| Vague quantification | "Much better"                    | Provide numbers                  |

---

## 7. The Pre-Commit Checklist

Before finalizing any documentation:

**Completeness**

- [ ] Would someone new understand WHY, not just WHAT?
- [ ] Are constraints that forced decisions explicit?
- [ ] Are tradeoffs acknowledged?

**Reasoning Quality**

- [ ] No circular reasoning?
- [ ] No unsupported assertions?
- [ ] Evidence level appropriate for decision importance?

**Precision**

- [ ] Numbers instead of vague quantifiers?
- [ ] Specific alternatives named, not just "other options"?

**Honesty**

- [ ] Limitations stated?
- [ ] Failed approaches included?
- [ ] Uncertainty appropriately hedged?

---

## 8. Adapting to Context

### 8.1 Time-Constrained Documentation

When time is short, prioritize:

1. **Critical decisions** (what would be hardest to reverse)
2. **Non-obvious choices** (what would surprise someone)
3. **Constraint-driven decisions** (what was forced, not chosen)

Skip:

- Obvious defaults
- Decisions that match well-known conventions
- Easily reversible choices

### 8.2 Audience Calibration

| Audience           | Emphasize                             | De-emphasize             |
| ------------------ | ------------------------------------- | ------------------------ |
| Professor/Reviewer | Reasoning, alternatives, evidence     | Implementation details   |
| Future maintainer  | Context, gotchas, reversal conditions | Background theory        |
| Teammate           | Decisions made, rationale             | Basics they already know |
| Report/Paper       | Methodology, reproducibility          | Internal tradeoffs       |

---

## 9. Question Bank Generator

To generate documentation questions for any new component, fill in this template:

```
Component: [NAME]
Type: [algorithm / architecture / data processing / configuration / tool choice]

Auto-generated questions:

1. What problem does [NAME] solve?
2. Why [NAME] instead of [most obvious alternative]?
3. What does [NAME] assume about the input/environment?
4. What happens if [NAME]'s assumptions are violated?
5. What would change if we had [2x resources / half the resources]?
6. What's the failure mode of [NAME]?
7. How do we know [NAME] is working correctly?
8. What would make us replace [NAME]?
```

Answer these, and you have solid documentation.

---

## 10. Summary: The Documentation Mindset

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   Every decision is a choice among alternatives.                │
│   Document the alternatives, the criteria, and the evidence.    │
│   Acknowledge what you traded away.                             │
│   State when you'd reverse the decision.                        │
│                                                                 │
│   The goal isn't to prove you're right.                         │
│   The goal is to show you thought carefully.                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Changelog

| Date       | Author | Change          |
| ---------- | ------ | --------------- |
| 2026-01-22 | Anand  | Initial version |
