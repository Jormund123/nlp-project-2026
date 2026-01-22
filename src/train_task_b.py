"""Minimal runnable training script for Task B (CodeBERT multiclass).

This script contains a small smoke-test training loop over a tiny synthetic
dataset so you can verify the end-to-end tokenization, forward, backward, and
checkpoint save steps work in your environment.
"""

import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer


class CodeBERTMultiClassifier(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.codebert = AutoModel.from_pretrained("microsoft/codebert-base")
        self.classifier = nn.Linear(768, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.codebert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0]
        return self.classifier(cls_output)


class SmallCodeDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt',
        )
        item = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(self.labels[idx], dtype=torch.long),
        }
        return item


def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--preprocess', action='store_true')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--save_dir', type=str, default='outputs/002-zeev-mgc-detection/models')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

    # Synthetic tiny dataset for smoke test
    texts = [
        "def add(a, b):\n    return a + b",
        "def sum_vals(a,b):\n    return a+b",
        "public void main(String[] args) { System.out.println(\"hi\"); }",
        "function avg(a,b){return (a+b)/2}",
    ]
    labels = [0, 0, 1, 1]  # synthetic multiclass labels (2 classes)

    dataset = SmallCodeDataset(texts, labels, tokenizer, max_length=128)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    n_classes = len(set(labels))
    model = CodeBERTMultiClassifier(n_classes=n_classes).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

    os.makedirs(args.save_dir, exist_ok=True)

    print("Starting smoke training: 1 epoch on tiny synthetic dataset")
    avg_loss = train_one_epoch(model, dataloader, optimizer, device)
    print(f"Finished epoch 1 — avg loss: {avg_loss:.4f}")

    save_path = os.path.join(args.save_dir, "model_task_b_epoch1.pt")
    torch.save({'model_state_dict': model.state_dict()}, save_path)
    print(f"Saved checkpoint to {save_path}")


if __name__ == '__main__':
    main()
