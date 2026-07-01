import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "synthetic_chat_dataset.json"
MODEL_SAVE_PATH = PROJECT_ROOT / "data" / "processed" / "student_model.pt"

BASE_MODEL = "Maltehb/danish-bert-botxo"
SEVERITY_ORDER = ["Ingen", "Lav", "Medium", "Høj"]
LABEL_TO_ID = {label: i for i, label in enumerate(SEVERITY_ORDER)}
NUM_CLASSES = len(SEVERITY_ORDER)

MAX_LENGTH = 256          # token limit per user (concatenated messages)
BATCH_SIZE = 8            # conservative for 8GB VRAM with BERT-base
EPOCHS = 4
LEARNING_RATE = 2e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

# Data loading
with open(DATASET_PATH, "r", encoding="utf-8") as f:
    raw_users = json.load(f)

# Concatenate each user's messages into a single text block.
# A separator token helps the model see message boundaries.
texts = [" [SEP] ".join(u["messages"]) for u in raw_users]
grimt_labels = [LABEL_TO_ID[u["grimt_sprog_severity"]] for u in raw_users]
ludomani_labels = [LABEL_TO_ID[u["ludomani_severity"]] for u in raw_users]

print(f"Loaded {len(texts)} users for training")

# Train / val / test split (stratify on grimt_sprog as the primary split key;
# not perfect for both labels simultaneously, but reasonable given joint training)
indices = list(range(len(texts)))
train_idx, temp_idx = train_test_split(
    indices, test_size=0.3, random_state=42, stratify=grimt_labels
)
val_idx, test_idx = train_test_split(
    temp_idx, test_size=0.5, random_state=42,
    stratify=[grimt_labels[i] for i in temp_idx]
)

print(f"Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")

# Dataset class
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)


class ChatDataset(Dataset):
    def __init__(self, idx_list):
        self.idx_list = idx_list

    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, i):
        idx = self.idx_list[i]
        encoding = tokenizer(
            texts[idx],
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "grimt_label": torch.tensor(grimt_labels[idx], dtype=torch.long),
            "ludomani_label": torch.tensor(ludomani_labels[idx], dtype=torch.long),
        }


train_loader = DataLoader(ChatDataset(train_idx), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(ChatDataset(val_idx), batch_size=BATCH_SIZE)
test_loader = DataLoader(ChatDataset(test_idx), batch_size=BATCH_SIZE)


# Model: shared encoder + two classification heads
class MultiTaskModerationModel(nn.Module):
    def __init__(self, base_model_name: str, num_classes: int):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.2)
        self.grimt_head = nn.Linear(hidden_size, num_classes)
        self.ludomani_head = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]  # [CLS] token representation
        pooled = self.dropout(pooled)
        grimt_logits = self.grimt_head(pooled)
        ludomani_logits = self.ludomani_head(pooled)
        return grimt_logits, ludomani_logits


model = MultiTaskModerationModel(BASE_MODEL, NUM_CLASSES).to(DEVICE)

# Training setup
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
)
criterion = nn.CrossEntropyLoss()


def run_epoch(loader, training: bool):
    model.train() if training else model.eval()
    total_loss = 0.0

    with torch.set_grad_enabled(training):
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            grimt_label = batch["grimt_label"].to(DEVICE)
            ludomani_label = batch["ludomani_label"].to(DEVICE)

            if training:
                optimizer.zero_grad()

            grimt_logits, ludomani_logits = model(input_ids, attention_mask)

            # Combined loss: simple equal-weighted sum of both task losses.
            # Worth revisiting if one task dominates the other during training.
            loss_grimt = criterion(grimt_logits, grimt_label)
            loss_ludomani = criterion(ludomani_logits, ludomani_label)
            loss = loss_grimt + loss_ludomani

            if training:
                loss.backward()
                optimizer.step()
                scheduler.step()

            total_loss += loss.item()

    return total_loss / len(loader)

# Training loop
print("\nStarting training...")
for epoch in range(EPOCHS):
    train_loss = run_epoch(train_loader, training=True)
    val_loss = run_epoch(val_loader, training=False)
    print(f"Epoch {epoch+1}/{EPOCHS} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

# Save the trained model
MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"\nModel saved to {MODEL_SAVE_PATH}")


# Evaluation on held-out test set
model.eval()
grimt_preds, grimt_true = [], []
ludomani_preds, ludomani_true = [], []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)

        grimt_logits, ludomani_logits = model(input_ids, attention_mask)

        grimt_preds.extend(torch.argmax(grimt_logits, dim=1).cpu().numpy())
        ludomani_preds.extend(torch.argmax(ludomani_logits, dim=1).cpu().numpy())
        grimt_true.extend(batch["grimt_label"].numpy())
        ludomani_true.extend(batch["ludomani_label"].numpy())

print("\n" + "=" * 60)
print("GRIMT SPROG — Test set performance")
print("=" * 60)
print(classification_report(grimt_true, grimt_preds, target_names=SEVERITY_ORDER, zero_division=0))

print("\n" + "=" * 60)
print("LUDOMANI — Test set performance")
print("=" * 60)
print(classification_report(ludomani_true, ludomani_preds, target_names=SEVERITY_ORDER, zero_division=0))