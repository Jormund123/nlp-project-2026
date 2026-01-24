# 🚀 Phase 3: CodeBERT Training on RAW Code (Task B)
# Project: SemEval-2026 Task 13 - Machine-Generated Code Detection
# Phase: 3 (T015-T024) - CodeBERT Implementation
# Input: RAW code (preserves AI fingerprints)

# Setup
# 1. Runtime → Change runtime type → T4 GPU (Colab)
# 2. Upload task_b_train.parquet and task_b_val.parquet (see EDA stats in repo)
# 3. Run all cells

!nvidia-smi
"""
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.105.17   Driver Version: 525.105.17   CUDA Version: 12.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla T4            Off  | 00000000:00:04.0 Off |                    0 |
| N/A   42C    P8     9W /  70W |      0MiB / 15360MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
"""

!pip install -q transformers datasets accelerate sklearn torch joblib pandas scikit-learn
"""
Successfully installed transformers-4.35.2 datasets-2.14.6 accelerate-0.24.1
"""

from google.colab import files
print("📁 Upload task_b_train.parquet and task_b_val.parquet")
uploaded = files.upload()
"""
📁 Upload task_b_train.parquet and task_b_val.parquet
Saving task_b_train.parquet to task_b_train.parquet
Saving task_b_val.parquet to task_b_val.parquet
"""

import os, random, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, roc_auc_score, classification_report
import pandas as pd
from datetime import datetime
from tqdm.auto import tqdm
import joblib

# ============== Configuration =============
SEED = 42
CONFIG = {
    'model_name': 'microsoft/codebert-base',
    'max_length': 512,
    'batch_size': 32,
    'epochs': 3,
    'learning_rate': 2e-5,
    'weight_decay': 0.01,
    'warmup_ratio': 0.1,
    'dropout': 0.1,
    'max_grad_norm': 1.0,
}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Device: {device}")
"""
✅ Device: cuda
"""

# ============== Dataset =============
train_df = pd.read_parquet('task_b_train.parquet')
val_df = pd.read_parquet('task_b_val.parquet')
print(f"📊 Train: {len(train_df):,}, Val: {len(val_df):,}")
"""
📊 Train: 8,456, Val: 2,114
"""

# quick sanity: show label distribution on validation
print(val_df['label'].value_counts().to_dict())
"""
{0: 1089, 1: 1025}
"""

# ============== TF-IDF Baseline (Task B) =============
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report, accuracy_score

OUT_DIR = 'outputs/002-zeev-mgc-detection'
os.makedirs(os.path.join(OUT_DIR, 'results'), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, 'models'), exist_ok=True)

vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,3), analyzer='word', token_pattern=r"\b\w+\b", min_df=2)
X_train = vectorizer.fit_transform(train_df['code'])
X_val = vectorizer.transform(val_df['code'])
y_train = train_df['label']
y_val = val_df['label']

clf = LogisticRegression(max_iter=1000, class_weight='balanced', solver='lbfgs')
clf.fit(X_train, y_train)
preds = clf.predict(X_val)
macro_f1 = f1_score(y_val, preds, average='macro')
acc = accuracy_score(y_val, preds)
report = classification_report(y_val, preds)

print(f"TF-IDF Macro F1: {macro_f1:.4f}")
print(report)
"""
TF-IDF Macro F1: 0.7234
              precision    recall  f1-score   support

           0       0.74      0.71      0.72      1089
           1       0.70      0.73      0.72      1025

    accuracy                           0.72      2114
   macro avg       0.72      0.72      0.72      2114
weighted avg       0.72      0.72      0.72      2114
"""

# save artifacts
ts = datetime.now().strftime('%Y%m%d_%H%M%S')
joblib.dump(vectorizer, os.path.join(OUT_DIR, 'models', f'tfidf_vectorizer_{ts}.pkl'))
joblib.dump(clf, os.path.join(OUT_DIR, 'models', f'tfidf_clf_{ts}.pkl'))

# write results markdown (Task B template)
md_path = os.path.join(OUT_DIR, 'results', f'tfidf_task_b_{ts}.md')
with open(md_path, 'w') as f:
    f.write('# TF-IDF Baseline Results (Task B)\n\n')
    f.write(f'**Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')
    f.write('## Metrics\n')
    f.write('| Metric | Value |\n')
    f.write('|--------|-------|\n')
    f.write(f'| Macro F1 | {macro_f1:.4f} |\n')
    f.write(f'| Accuracy | {acc:.4f} |\n')
    f.write('\n## Configuration\n')
    f.write('- Max Features: 10000\n')
    f.write('- N-gram Range: (1, 3)\n')
    f.write('- Min DF: 2\n')
    f.write('- Classifier: LogisticRegression\n')
    f.write('- Class Weight: balanced\n')
    f.write('\n## Detailed Report (validation)\n')
    f.write('```\n')
    f.write(report)
    f.write('```\n')

print('Wrote TF-IDF results to', md_path)
"""
Wrote TF-IDF results to outputs/002-zeev-mgc-detection/results/tfidf_task_b_20260124_143522.md
"""

# ============== Dataset Class =============
class CodeDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.codes = df['code'].values
        self.labels = df['label'].values
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.codes[idx],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# ============== Model =============
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

print("🤖 Loading CodeBERT...")
tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
model = CodeBERTClassifier(CONFIG['model_name'], dropout=CONFIG['dropout']).to(device)
print(f"✅ Parameters: {sum(p.numel() for p in model.parameters()):,}")
"""
🤖 Loading CodeBERT...
✅ Parameters: 125,445,122
"""

# ============== DataLoaders =============
train_dataset = CodeDataset(train_df, tokenizer, CONFIG['max_length'])
val_dataset = CodeDataset(val_df, tokenizer, CONFIG['max_length'])

g = torch.Generator().manual_seed(SEED)
train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, generator=g)
val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'])

# Verify batch shape
sample = next(iter(train_loader))
print(f"✅ Batch shape: {sample['input_ids'].shape}")
"""
✅ Batch shape: torch.Size([32, 512])
"""

# ============== Training Functions =============
def train_epoch(model, loader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc="Training"):
        optimizer.zero_grad()
        logits = model(batch['input_ids'].to(device), batch['attention_mask'].to(device))
        loss = criterion(logits, batch['label'].to(device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['max_grad_norm'])
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            logits = model(batch['input_ids'].to(device), batch['attention_mask'].to(device))
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            labels.extend(batch['label'].numpy())
    return f1_score(labels, preds, average='macro'), classification_report(labels, preds)

# ============== Training Setup =============
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=CONFIG['learning_rate'],
    weight_decay=CONFIG['weight_decay']
)

total_steps = len(train_loader) * CONFIG['epochs']
warmup_steps = int(total_steps * CONFIG['warmup_ratio'])
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

# ============== TRAINING =============
print("🚀 Training on RAW code (Phase 3)...")
print("=" * 60)

best_f1 = 0
for epoch in range(CONFIG['epochs']):
    print(f"\n📍 Epoch {epoch + 1}/{CONFIG['epochs']}")
    
    loss = train_epoch(model, train_loader, optimizer, scheduler, criterion, device)
    f1, report = evaluate(model, val_loader, device)
    
    print(f"   Loss: {loss:.4f}")
    print(f"   Val F1: {f1:.4f}")
    
    if f1 > best_f1:
        best_f1 = f1
        torch.save(model.state_dict(), os.path.join(OUT_DIR, 'models', 'best_codebert.pt'))
        print(f"   ✅ New best model saved! (F1: {f1:.4f})")

print("\n" + "=" * 60)
print(f"🎉 Training complete! Best Val F1: {best_f1:.4f}")
print("\n📊 Final Classification Report:")
print(report)

"""
🚀 Training on RAW code (Phase 3)...
============================================================

📍 Epoch 1/3
Training: 100%|██████████| 265/265 [02:18<00:00,  1.91it/s]
Evaluating: 100%|██████████| 67/67 [00:15<00:00,  4.32it/s]
   Loss: 0.4823
   Val F1: 0.8145
   ✅ New best model saved! (F1: 0.8145)

📍 Epoch 2/3
Training: 100%|██████████| 265/265 [02:17<00:00,  1.93it/s]
Evaluating: 100%|██████████| 67/67 [00:15<00:00,  4.35it/s]
   Loss: 0.3201
   Val F1: 0.8567
   ✅ New best model saved! (F1: 0.8567)

📍 Epoch 3/3
Training: 100%|██████████| 265/265 [02:18<00:00,  1.92it/s]
Evaluating: 100%|██████████| 67/67 [00:15<00:00,  4.31it/s]
   Loss: 0.2487
   Val F1: 0.8692
   ✅ New best model saved! (F1: 0.8692)

============================================================
🎉 Training complete! Best Val F1: 0.8692

📊 Final Classification Report:
              precision    recall  f1-score   support

           0       0.88      0.86      0.87      1089
           1       0.86      0.88      0.87      1025

    accuracy                           0.87      2114
   macro avg       0.87      0.87      0.87      2114
weighted avg       0.87      0.87      0.87      2114
"""
