"""
TF-IDF Baseline for Machine-Generated Code Detection (Task A)

This script implements the lexical baseline:
1. Loads stratified training/validation data
2. Extracts TF-IDF features (word-level, trigrams)
3. Trains Logistic Regression classifier
4. Evaluates and saves results to outputs/
"""

import pandas as pd
import os
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime

# Project configuration
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
OUTPUTS_DIR = os.path.join(ROOT_DIR, "outputs")
STATS_DIR = os.path.join(OUTPUTS_DIR, "statistics")
RESULTS_DIR = os.path.join(OUTPUTS_DIR, "results")
MODELS_DIR = os.path.join(OUTPUTS_DIR, "models")
FIGURES_DIR = os.path.join(OUTPUTS_DIR, "figures")

# Create directories
for d in [STATS_DIR, RESULTS_DIR, MODELS_DIR, FIGURES_DIR]:
    os.makedirs(d, exist_ok=True)

# Visualization settings
plt.style.use('seaborn-v0_8-whitegrid')
FIGURE_DPI = 300

def train_tfidf_baseline():
    print(f"\n{'-'*50}")
    print("Starting TF-IDF Baseline Training (Task A)")
    print(f"{'-'*50}")

    # 1. Load Data
    # NOTE: Always use RAW 'code' column for primary experiments per project rules
    print("Loading datasets...")
    train_df = pd.read_parquet(os.path.join(DATA_DIR, "task_a_train.parquet"))
    val_df = pd.read_parquet(os.path.join(DATA_DIR, "task_a_val.parquet"))
    
    # Validation check
    assert 'code' in train_df.columns, "Missing 'code' column"
    assert 'label' in train_df.columns, "Missing 'label' column"
    
    print(f"Train size: {len(train_df)}")
    print(f"Val size:   {len(val_df)}")

    # 2. Extract Features
    print("\nExtracting TF-IDF features...")
    start_time = time.time()
    
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 3),      # Trigrams per research.md
        analyzer='word',
        token_pattern=r'\b\w+\b',
        min_df=2,
    )
    
    # Fit on train, transform both
    X_train = vectorizer.fit_transform(train_df['code'])
    X_val = vectorizer.transform(val_df['code'])
    y_train = train_df['label']
    y_val = val_df['label']
    
    print(f"Feature extraction took {time.time() - start_time:.1f}s")
    print(f"Vocab size: {len(vectorizer.vocabulary_)}")

    # 3. Train Model
    print("\nTraining Logistic Regression...")
    clf = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        solver='lbfgs',
        random_state=42,
    )
    
    clf.fit(X_train, y_train)
    print("Training complete.")

    # 4. Evaluate
    print("\nEvaluating...")
    val_preds = clf.predict(X_val)
    val_probs = clf.predict_proba(X_val)[:, 1]
    
    # Metrics
    f1 = f1_score(y_val, val_preds)
    roc_auc = roc_auc_score(y_val, val_probs)
    report = classification_report(y_val, val_preds)
    
    print(f"\nResults:")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print("\nClassification Report:\n", report)

    # 5. Save Artifacts
    
    # Save Metrics to Markdown
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    results_path = os.path.join(RESULTS_DIR, "tfidf_baseline_task_a.md")
    
    with open(results_path, 'w') as f:
        f.write(f"# TF-IDF Baseline Results (Task A)\n\n")
        f.write(f"**Generated**: {timestamp}\n\n")
        f.write("## Metrics\n")
        f.write(f"| Metric | Value |\n")
        f.write(f"|--------|-------|\n")
        f.write(f"| **F1** | **{f1:.4f}** |\n")
        f.write(f"| ROC-AUC | {roc_auc:.4f} |\n\n")
        f.write("## Configuration\n")
        f.write("- `max_features`: 10000\n")
        f.write("- `ngram_range`: (1, 3)\n")
        f.write("- `min_df`: 2\n")
        f.write("- `model`: LogisticRegression(balanced)\n\n")
        f.write("## Detailed Report\n")
        f.write("```\n")
        f.write(report)
        f.write("\n```\n")
        
    print(f"✓ Results saved to {results_path}")

    # Save Model
    model_path = os.path.join(MODELS_DIR, "tfidf_logreg_task_a.joblib")
    joblib.dump({'model': clf, 'vectorizer': vectorizer}, model_path)
    print(f"✓ Model saved to {model_path}")

    # Generate Confusion Matrix
    cm = confusion_matrix(y_val, val_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'TF-IDF Baseline (F1={f1:.3f})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    fig_path = os.path.join(FIGURES_DIR, "confusion_matrix_tfidf.png")
    plt.savefig(fig_path, dpi=FIGURE_DPI)
    print(f"✓ Confusion matrix saved to {fig_path}")

    return f1

if __name__ == "__main__":
    train_tfidf_baseline()
