"""
Generate Kaggle Submission for Task A (Binary Classification)

Usage:
    python src/generate_submission.py
"""

import os
import torch
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
from tqdm import tqdm

# Config
MODEL_NAME = "microsoft/codebert-base"
MODEL_PATH = "outputs/models/model_task_a_raw.pt"
OUTPUT_FILE = "outputs/submissions/submission_task_a.csv"
BATCH_SIZE = 32
MAX_LENGTH = 512

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.backends.mps.is_available():
    device = torch.device("mps")

print(f"Using device: {device}")

# Model Definition (Must match training)
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

def generate_submission():
    # 1. Load Model
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found at {MODEL_PATH}")
        print("Please download 'model_task_a_raw.pt' from Colab and place it in outputs/models/")
        return

    print(f"Loading model from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = CodeBERTClassifier(MODEL_NAME).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # 2. Load Test Data
    print("Loading test dataset from HuggingFace...")
    # Using the official dataset ID
    dataset = load_dataset("DaniilOr/SemEval-2026-Task13", split="test")
    # Filter for Task A if necessary, or assuming test set structure.
    # The dataset typically has 'id', 'text' (or 'code').
    # Let's check columns during runtime if possible, but for now assume 'text' or 'code'
    
    print(f"Test samples: {len(dataset)}")
    
    # 3. Predict
    all_preds = []
    ids = []
    
    # Process in batches
    for i in tqdm(range(0, len(dataset), BATCH_SIZE), desc="Predicting"):
        batch = dataset[i : i + BATCH_SIZE]
        ids.extend(batch['id'])
        
        # Handle column name variation
        code_col = 'code' if 'code' in batch else 'text'
        codes = batch[code_col]
        
        encoding = tokenizer(
            codes,
            max_length=MAX_LENGTH,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        with torch.no_grad():
            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())

    # 4. Save
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df = pd.DataFrame({
        'id': ids,
        'label': all_preds
    })
    
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ Submission saved to {OUTPUT_FILE}")
    print(f"Total predictions: {len(df)}")
    print(df.head())

if __name__ == "__main__":
    generate_submission()
