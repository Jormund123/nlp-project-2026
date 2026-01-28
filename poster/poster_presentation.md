# Poster Presentation Speaker Notes: SemEval-2026 Task 13

**Speakers**: Anand Karna (Task A) & Zeev [Lastname] (Task B)

---

## 1. The "Fingerprint Effect" Hypothesis (Column 1)

**Speaker: Anand**

Logic is universal, but style is personal. Our central thesis is that in the domain of Machine-Generated Code detection, the _style_—whitespace, indentation, variable naming conventions—is a significantly stronger signal than the actual algorithmic logic.

This is based on the concept of Entropy. Humans are high-entropy writers; we are chaotic. We mix tabs and spaces, use inconsistent variable names, and write informal comments. AI models, on the other hand, are low-entropy engines. They are probabilistic machines driven to the "average," defaulting to perfect PEP-8 formatting and standard docstrings every time.

The "trap" in this field is that standard NLP practices—like lowercasing text or removing punctuation—destroy this entropy. If you clean human code, you inadvertently stick it into the "perfect" distribution of AI code, erasing the very signal you are trying to detect. That is why we avoided standard preprocessing.

## 2. Architectural Decisions (Column 1)

**Speaker: Anand**

For our baseline, we implemented a simple TF-IDF vectorizer in `baseline_tfidf.py`. We used 10,000 features and 1-3 ngrams feeding into a Logistic Regression model. This demonstrated that simple keyword counting—like finding the word "generated" or "solution"—can achieve 83% accuracy without any understanding of syntax.

For our primary model, implemented in `train_task_a.py`, we chose `microsoft/codebert-base`. Unlike standard BERT, CodeBERT is pre-trained on code and understands syntax like indentation and brackets. We added a custom linear classification head on top. Crucially, we built a custom `CodeDataset` class that accepts **RAW** strings, explicitly disabling the `normalize_whitespace` function to preserve those all-important stylistic fingerprints.

## 3. Data Strategy (Column 1)

**Speaker: Anand**

We faced a constraint where we had 500k samples but could only use 20k because our Colab environment (12GB RAM) would crash with the full dataset. To handle this, we didn't just split the data randomly.

We wrote a custom stratified splitter that stratified by **Label AND Language**. This was critical to stop the model from learning shortcuts. If we had 1000 AI Java files and only 10 Human Java files, the model would simply learn that "Java equals AI." Our stratification strategy prevents this bias and forces the model to learn actual distinctions.

## 4. Task A (Binary): Results & Ablation (Column 2)

**Speaker: Anand**

Our results were definitive. The TF-IDF baseline achieved an F1 of 83%, but our CodeBERT model on raw code reached 98.5%—a massive improvement.

To prove our thesis, we conducted an **Ablation Study**. An ablation study is like a scientific control where we deliberately "break" one part of the system to see if it matters. We trained a second model on **Preprocessed Code**, where we stripped all comments and normalized whitespace.

The result was striking: the error rate nearly **doubled**. This confirms that for the hardest 1.4% of cases, the model relies almost entirely on stylistic cues (the "chaos" of human formatting) to make the decision. When we removed the style, we lost the signal.

## 5. Theoretical Foundations (Column 2)

**Speaker: Zeev**

The reason CodeBERT outperforms TF-IDF so significantly is **Self-Attention**. TF-IDF simply counts isolated words, telling you "there are 5 `if` statements." CodeBERT sees the relationships. It understands that a `return` statement depends on a variable defined 50 lines up. It captures the structure of the code, not just the vocabulary.

## 6. Task B (Multi-Class): Attribution (Column 3)

**Speaker: Zeev**

Task A is like distinguishing a Bird from a Plane. Task B is harder: it's like distinguishing a Boeing 737 from an Airbus A320. All these LLMs (GPT-4, Llama-2, StarCoder) are trained on similar data and write correct code, making them semantically nearly identical.

We adapted our classifier in `train_task_b.py` for 11 classes and implemented a **Macro F1** metric callback. This ensures that accurate detection of small AI models (like `phi-1`) counts just as much as detecting the huge Human class.

Our raw CodeBERT model achieved a 72% F1 score. The impact of style was even more pronounced here. When we stripped style in the ablation study, performance dropped by 3.6%—three times worse than in Task A. This suggests that distinguishing between two AIs relies almost entirely on subtle formatting quirks, like how they format a list comprehension.

## 7. Limitations & Ethics (Column 3)

**Speaker: Zeev/Anand**

We must acknowledge that 90% of our data is Python, so we cannot guarantee these results transfer to C++ or Java.

More importantly, regarding ethics: even with a 98% accuracy, we have a 2% False Positive rate. In a lecture hall of 200 students, this model would falsely accuse 4 innocent people. Therefore, our verdict is that this tool must be used only as a **screening aid** to flag files for human review. It should never be used as an automated judge to fail a student.

## 8. Closing

**Speaker: Zeev/Anand**

We didn't just build a detector; we proved that machine code has a unique timestamp—a stylistic fingerprint—that sets it apart from the chaos of human creativity. Thank you.
