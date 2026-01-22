"""TF-IDF baseline for Task B (multiclass per-generator).

Implements train_tfidf_baseline(train_texts, train_labels, val_texts, val_labels)
and a small CLI that runs on synthetic data when executed directly.
"""
import argparse
import os
from typing import List, Tuple

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_fscore_support


def train_tfidf_baseline(
    train_texts: List[str],
    train_labels: List[int],
    val_texts: List[str],
    val_labels: List[int],
    output_dir: str,
    max_features: int = 10000,
    ngram_range: Tuple[int, int] = (1, 3),
):
    os.makedirs(output_dir, exist_ok=True)

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        analyzer="word",
        token_pattern=r"\b\w+\b",
        min_df=2,
    )

    X_train = vectorizer.fit_transform(train_texts)
    X_val = vectorizer.transform(val_texts)

    clf = LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs")
    clf.fit(X_train, train_labels)

    preds = clf.predict(X_val)
    macro_f1 = f1_score(val_labels, preds, average="macro")
    per_label = precision_recall_fscore_support(val_labels, preds, average=None)

    # Save artifacts
    joblib.dump(vectorizer, os.path.join(output_dir, "tfidf_vectorizer.pkl"))
    joblib.dump(clf, os.path.join(output_dir, "tfidf_clf.pkl"))

    return {
        "macro_f1": float(macro_f1),
        "per_label": [float(x) for x in per_label[2].tolist()],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="outputs/002-zeev-mgc-detection/models")
    args = parser.parse_args()

    # Synthetic small dataset for smoke test
    train_texts = [
        "def add(a, b):\n    return a + b",
        "// compute average\nfunction avg(a,b){return (a+b)/2}",
        "def sort_list(x):\n    return sorted(x)",
        "public void main(String[] args) { System.out.println(\"hi\"); }",
    ]
    train_labels = [0, 1, 0, 2]  # synthetic generator labels

    val_texts = [
        "def sum_vals(a,b):\n    return a+b",
        "function max(a,b){return a>b?a:b}",
    ]
    val_labels = [0, 1]

    results = train_tfidf_baseline(train_texts, train_labels, val_texts, val_labels, args.output_dir)
    print("TF-IDF baseline smoke test results:", results)
