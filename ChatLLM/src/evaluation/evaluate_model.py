import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # ChatLLM/
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "synthetic_chat_dataset.json"
MODEL_PATH = PROJECT_ROOT / "data" / "processed" / "student_model.pt"
DOCS_PATH = PROJECT_ROOT / "docs"
DOCS_PATH.mkdir(parents=True, exist_ok=True)

BASE_MODEL = "Maltehb/danish-bert-botxo"
SEVERITY_ORDER = ["Ingen", "Lav", "Medium", "Høj"]
LABEL_TO_ID = {label: i for i, label in enumerate(SEVERITY_ORDER)}
NUM_CLASSES = len(SEVERITY_ORDER)

MAX_LENGTH = 256
BATCH_SIZE = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

# Reload data + recreate the exact same test split (same random_state=42)

with open(DATASET_PATH, "r", encoding="utf-8") as f:
    raw_users = json.load(f)

texts = [" [SEP] ".join(u["messages"]) for u in raw_users]
grimt_labels = [LABEL_TO_ID[u["grimt_sprog_severity"]] for u in raw_users]
ludomani_labels = [LABEL_TO_ID[u["ludomani_severity"]] for u in raw_users]
user_ids = [u["user_id"] for u in raw_users]

indices = list(range(len(texts)))
train_idx, temp_idx = train_test_split(
    indices, test_size=0.3, random_state=42, stratify=grimt_labels
)
val_idx, test_idx = train_test_split(
    temp_idx, test_size=0.5, random_state=42,
    stratify=[grimt_labels[i] for i in temp_idx]
)

print(f"Test set: {len(test_idx)} users")

# Dataset / model definitions — identical to training script

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
            "original_idx": idx,
        }


test_loader = DataLoader(ChatDataset(test_idx), batch_size=BATCH_SIZE)


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
        pooled = outputs.last_hidden_state[:, 0, :]
        pooled = self.dropout(pooled)
        return self.grimt_head(pooled), self.ludomani_head(pooled)


model = MultiTaskModerationModel(BASE_MODEL, NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print("Model loaded.")

# Run predictions on test set, keeping track of original_idx for error lookup

grimt_preds, grimt_true = [], []
ludomani_preds, ludomani_true = [], []
all_original_idx = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)

        grimt_logits, ludomani_logits = model(input_ids, attention_mask)

        grimt_preds.extend(torch.argmax(grimt_logits, dim=1).cpu().numpy())
        ludomani_preds.extend(torch.argmax(ludomani_logits, dim=1).cpu().numpy())
        grimt_true.extend(batch["grimt_label"].numpy())
        ludomani_true.extend(batch["ludomani_label"].numpy())
        all_original_idx.extend(batch["original_idx"].numpy())

# Confusion matrices

def plot_confusion(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred, labels=range(NUM_CLASSES))
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=SEVERITY_ORDER, yticklabels=SEVERITY_ORDER,
        cbar_kws={"label": "Antal brugere"}
    )
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig(DOCS_PATH / filename, dpi=150)
    plt.show()
    return cm

print("\nGenerating confusion matrices...")
grimt_cm = plot_confusion(grimt_true, grimt_preds, "Grimt sprog — Confusion Matrix", "grimt_confusion_matrix.png")
ludomani_cm = plot_confusion(ludomani_true, ludomani_preds, "Ludomani — Confusion Matrix", "ludomani_confusion_matrix.png")

print("\nGrimt sprog confusion matrix (rows=true, cols=predicted):")
print(SEVERITY_ORDER)
print(grimt_cm)

print("\nLudomani confusion matrix (rows=true, cols=predicted):")
print(SEVERITY_ORDER)
print(ludomani_cm)

# Pull actual misclassified examples for manual error analysis

def show_misclassified(true_labels, pred_labels, label_name, n=8):
    print(f"\n{'=' * 70}")
    print(f"MISCLASSIFIED EXAMPLES — {label_name}")
    print(f"{'=' * 70}")

    mismatches = [
        (i, t, p) for i, (t, p) in enumerate(zip(true_labels, pred_labels)) if t != p
    ]
    print(f"Total mismatches: {len(mismatches)} / {len(true_labels)}")

    # Prioritize the most severe misses: true Høj predicted as Ingen/Lav (dangerous false negatives)
    # and true Ingen predicted as Høj (costly false positives)
    severe_misses = [
        (i, t, p) for (i, t, p) in mismatches
        if abs(t - p) >= 2  # jumped at least 2 severity levels
    ]
    to_show = severe_misses[:n] if severe_misses else mismatches[:n]

    for i, t, p in to_show:
        original_idx = all_original_idx[i]
        user = raw_users[original_idx]
        print(f"\nuser_id={user['user_id']} | true={SEVERITY_ORDER[t]} -> predicted={SEVERITY_ORDER[p]}")
        for m in user["messages"][:3]:
            print(f"  - {m}")


show_misclassified(grimt_true, grimt_preds, "Grimt sprog")
show_misclassified(ludomani_true, ludomani_preds, "Ludomani")

print("\nDone. Confusion matrices saved to docs/")