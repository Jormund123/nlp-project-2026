"""
CodeBERT Training Script for Task A (Binary Classification)

Usage:
    python src/train_task_a.py              # Train on raw code (default)
    python src/train_task_a.py --preprocess # Train on preprocessed code (ablation)
"""

import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, roc_auc_score, classification_report
import pandas as pd
from datetime import datetime
from tqdm import tqdm

# ============== Configuration ==============
SEED = 42
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
OUTPUTS_DIR = os.path.join(ROOT_DIR, "outputs")
MODELS_DIR = os.path.join(OUTPUTS_DIR, "models")
RESULTS_DIR = os.path.join(OUTPUTS_DIR, "results")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

CONFIG = {
    'model_name': 'microsoft/codebert-base',
    'max_length': 512,
    'batch_size': 8,           # Reduced for MPS memory
    'accum_steps': 4,          # Gradient accumulation (effective batch = 8*4=32)
    'epochs': 3,
    'learning_rate': 2e-5,
    'weight_decay': 0.01,
    'warmup_ratio': 0.1,
    'dropout': 0.1,
    'max_grad_norm': 1.0,
}


def set_seed(seed):
    """Set all random seeds for reproducibility (Audit 2.4)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ============== Dataset ==============
class CodeDataset(Dataset):
    def __init__(self, df, tokenizer, max_length, use_preprocessed=False):
        self.codes = df['code_preprocessed' if use_preprocessed else 'code'].tolist()
        self.labels = df['label'].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.codes)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.codes[idx],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }


# ============== Model ==============
class CodeBERTClassifier(nn.Module):
    def __init__(self, model_name, num_classes=2, dropout=0.1):
        super().__init__()
        self.codebert = AutoModel.from_pretrained(model_name)
        hidden_size = self.codebert.config.hidden_size  # Dynamic, not hardcoded 768
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.codebert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0]  # [CLS] token
        cls_output = self.dropout(cls_output)
        return self.classifier(cls_output)


# ============== Training ==============
def train_epoch(model, loader, optimizer, scheduler, criterion, device, max_grad_norm, accum_steps):
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    
    for step, batch in enumerate(tqdm(loader, desc="Training")):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels) / accum_steps  # Scale loss
        loss.backward()
        
        # Step optimizer every accum_steps
        if (step + 1) % accum_steps == 0 or (step + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accum_steps  # Unscale for logging
    
    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            logits = model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    f1 = f1_score(all_labels, all_preds)
    roc_auc = roc_auc_score(all_labels, all_probs)
    report = classification_report(all_labels, all_preds)
    
    return f1, roc_auc, report, all_preds


def main(use_preprocessed=False):
    # Reproducibility
    set_seed(SEED)
    
    device = get_device()
    print(f"Using device: {device}")
    
    variant = "preprocessed" if use_preprocessed else "raw"
    print(f"\n{'='*50}")
    print(f"Training CodeBERT on {variant.upper()} code")
    print(f"{'='*50}")
    
    # Load data
    print("Loading data...")
    train_df = pd.read_parquet(os.path.join(DATA_DIR, "task_a_train.parquet"))
    val_df = pd.read_parquet(os.path.join(DATA_DIR, "task_a_val.parquet"))
    print(f"Train: {len(train_df)}, Val: {len(val_df)}")
    
    # Tokenizer & Datasets
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
    train_dataset = CodeDataset(train_df, tokenizer, CONFIG['max_length'], use_preprocessed)
    val_dataset = CodeDataset(val_df, tokenizer, CONFIG['max_length'], use_preprocessed)
    
    # Seeded generator for reproducible shuffling
    g = torch.Generator()
    g.manual_seed(SEED)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=CONFIG['batch_size'], 
        shuffle=True,
        generator=g
    )
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'])
    
    # Verify batch shape
    sample_batch = next(iter(train_loader))
    print(f"Batch shape: {sample_batch['input_ids'].shape}")  # Should be (8, 512)
    
    # Model
    model = CodeBERTClassifier(
        CONFIG['model_name'], 
        dropout=CONFIG['dropout']
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )
    
    num_update_steps_per_epoch = (len(train_loader) + CONFIG['accum_steps'] - 1) // CONFIG['accum_steps']
    total_steps = num_update_steps_per_epoch * CONFIG['epochs']
    warmup_steps = int(CONFIG['warmup_ratio'] * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    
    # Training loop with best checkpoint tracking
    epoch_losses = []
    best_f1 = 0.0
    best_state = None
    
    for epoch in range(CONFIG['epochs']):
        print(f"\nEpoch {epoch + 1}/{CONFIG['epochs']}")
        loss = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, 
            device, CONFIG['max_grad_norm'], CONFIG['accum_steps']
        )
        epoch_losses.append(loss)
        print(f"Loss: {loss:.4f}")
        
        # Evaluate after each epoch
        f1, roc_auc, _, _ = evaluate(model, val_loader, device)
        print(f"Val F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")
        
        # Track best model
        if f1 > best_f1:
            best_f1 = f1
            best_state = model.state_dict().copy()
            print(f"  ↑ New best F1!")
    
    # Load best model for final evaluation
    if best_state:
        model.load_state_dict(best_state)
    
    # Final evaluation
    print("\nFinal Evaluation (best checkpoint)...")
    f1, roc_auc, report, _ = evaluate(model, val_loader, device)
    print(f"\nFinal Results:")
    print(f"F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")
    print(report)
    
    # Save best model
    model_path = os.path.join(MODELS_DIR, f"model_task_a_{variant}.pt")
    torch.save(best_state, model_path)
    print(f"✓ Model saved to {model_path}")
    
    # Save results
    results_path = os.path.join(RESULTS_DIR, f"codebert_{variant}_task_a.md")
    with open(results_path, 'w') as f:
        f.write(f"# CodeBERT Results - {variant.upper()} Code (Task A)\n\n")
        f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Metrics\n")
        f.write(f"| Metric | Value |\n|--------|-------|\n")
        f.write(f"| **F1** | **{f1:.4f}** |\n")
        f.write(f"| ROC-AUC | {roc_auc:.4f} |\n\n")
        f.write("## Configuration\n")
        f.write(f"- Seed: {SEED}\n")
        f.write(f"- Epochs: {CONFIG['epochs']}\n")
        f.write(f"- Batch size: {CONFIG['batch_size']}\n")
        f.write(f"- Learning rate: {CONFIG['learning_rate']}\n")
        f.write(f"- Dropout: {CONFIG['dropout']}\n\n")
        f.write("## Training Losses\n")
        for i, loss in enumerate(epoch_losses):
            f.write(f"- Epoch {i+1}: {loss:.4f}\n")
        f.write(f"\n## Report\n```\n{report}\n```\n")
    print(f"✓ Results saved to {results_path}")
    
    return f1, roc_auc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--preprocess', action='store_true', help='Use preprocessed code (ablation)')
    args = parser.parse_args()
    main(use_preprocessed=args.preprocess)
