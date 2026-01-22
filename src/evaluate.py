"""
Evaluation Script for Task 13 (MGC Detection)
Handles model loading, metric computation, and visualization generation.
"""

import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, roc_auc_score, classification_report, confusion_matrix, precision_recall_curve
from transformers import AutoTokenizer, AutoModel
import joblib
import torch.nn as nn

# Visualization Config
plt.style.use('seaborn-v0_8-whitegrid')
FIGURE_DPI = 300
FIGURE_SIZE = (8, 6)
FONT_SIZE = 12
COLORS = {'tfidf': '#1f77b4', 'codebert': '#ff7f0e', 'raw': '#2ca02c', 'preprocessed': '#7f7f7f'}

# Paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
OUTPUTS_DIR = os.path.join(ROOT_DIR, "outputs")
MODELS_DIR = os.path.join(OUTPUTS_DIR, "models")
FIGURES_DIR = os.path.join(OUTPUTS_DIR, "figures")

os.makedirs(FIGURES_DIR, exist_ok=True)

# Model Definition (Must match training script)
class CodeBERTClassifier(nn.Module):
    def __init__(self, model_name, num_classes=2, dropout=0.1):
        super().__init__()
        self.codebert = AutoModel.from_pretrained(model_name)
        hidden_size = self.codebert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.codebert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0]
        return self.classifier(self.dropout(cls_output))

def load_codebert_model(model_path, device):
    """Load trained CodeBERT model."""
    model_name = 'microsoft/codebert-base'
    model = CodeBERTClassifier(model_name).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def get_codebert_preds(model, tokenizer, codes, device, batch_size=32):
    """Generate predictions using CodeBERT."""
    all_preds = []
    all_probs = []
    
    # Simple batching
    for i in range(0, len(codes), batch_size):
        batch_codes = codes[i:i+batch_size]
        encoding = tokenizer(
            batch_codes,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        with torch.no_grad():
            logits = model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)
            
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        
    return np.array(all_preds), np.array(all_probs)

def evaluate_model(y_true, y_pred, y_prob=None, model_name="Model"):
    """Generic evaluation function (T025)."""
    f1 = f1_score(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    
    print(f"\n{'='*20} {model_name} Results {'='*20}")
    print(f"F1 Score: {f1:.4f}")
    if y_prob is not None:
        roc_auc = roc_auc_score(y_true, y_prob)
        print(f"ROC-AUC: {roc_auc:.4f}")
    print("\nClassification Report:")
    print(report)
    
    return f1, report

def compute_per_language_f1(df, y_pred):
    """Compute F1 score per language (T036)."""
    df['pred'] = y_pred
    results = []
    
    print("\nPer-Language Analysis:")
    print(f"{'Language':<15} {'Samples':<10} {'F1 Score':<10}")
    print("-" * 35)
    
    for lang in df['language'].unique():
        subset = df[df['language'] == lang]
        f1 = f1_score(subset['label'], subset['pred'])
        results.append({'language': lang, 'samples': len(subset), 'f1': f1})
        print(f"{lang:<15} {len(subset):<10} {f1:.4f}")
        
    return pd.DataFrame(results)

def generate_confusion_matrix(y_true, y_pred, title, filename):
    """Generate and save confusion matrix (T041)."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=FIGURE_SIZE)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, filename), dpi=FIGURE_DPI)
    plt.close()
    print(f"✓ Saved {filename}")

def generate_pr_curve(y_true, y_prob, title, filename):
    """Generate and save PR curve (T047)."""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    plt.figure(figsize=FIGURE_SIZE)
    plt.plot(recall, precision, marker='.', label=title)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve: {title}')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(FIGURES_DIR, filename), dpi=FIGURE_DPI)
    plt.close()
    print(f"✓ Saved {filename}")

if __name__ == "__main__":
    # Example usage for verification
    val_df = pd.read_parquet(os.path.join(DATA_DIR, "task_a_val.parquet"))
    
    # Load TF-IDF (Local)
    tfidf_path = os.path.join(MODELS_DIR, "tfidf_task_a.joblib")
    lr_path = os.path.join(MODELS_DIR, "logreg_task_a.joblib")
    
    if os.path.exists(tfidf_path) and os.path.exists(lr_path):
        print("Evaluating TF-IDF Baseline...")
        tfidf = joblib.load(tfidf_path)
        clf = joblib.load(lr_path)
        
        X_test = tfidf.transform(val_df['code'])
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]
        
        evaluate_model(val_df['label'], y_pred, y_prob, "TF-IDF Baseline")
        generate_confusion_matrix(val_df['label'], y_pred, "TF-IDF Confusion Matrix", "confusion_matrix_tfidf.png")
