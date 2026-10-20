
from pathlib import Path
import json
import random

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW

from src.dataset import create_dataloaders
from src.model import create_model, CLASS_NAMES


# ---------------- Configuration ----------------

SEED = 42
EPOCHS = 3          # Start small on CPU
LEARNING_RATE = 1e-3

ROOT_DIR = Path(__file__).resolve().parent
MODEL_DIR = ROOT_DIR / "models"
REPORT_DIR = ROOT_DIR / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = MODEL_DIR / "efficientnet_b0_best.pth"
HISTORY_PATH = REPORT_DIR / "training_history.json"


# ---------------- Reproducibility ----------------

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)
print("Classes:", CLASS_NAMES)


# ---------------- Data ----------------

train_loader, val_loader, _ = create_dataloaders(
    root_dir=ROOT_DIR
)

print("Training images:", len(train_loader.dataset))
print("Validation images:", len(val_loader.dataset))


# ---------------- Model ----------------

model = create_model(
    num_classes=len(CLASS_NAMES),
    freeze_features=True,
).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE,
    weight_decay=1e-4,
)


# ---------------- Evaluation function ----------------

def evaluate(model, loader):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item() * images.size(0)

            predictions = logits.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    average_loss = total_loss / total
    accuracy = correct / total

    return average_loss, accuracy


# ---------------- Training loop ----------------

best_val_accuracy = -1.0
history = []

for epoch in range(EPOCHS):
    model.train()

    # Keep frozen feature extractor and its BatchNorm
    # statistics in evaluation mode.
    model.features.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        predictions = logits.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / total
    train_accuracy = correct / total

    val_loss, val_accuracy = evaluate(model, val_loader)

    epoch_result = {
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
    }

    history.append(epoch_result)

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_accuracy:.4f}"
    )

    # Save the best checkpoint using validation accuracy.
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "class_names": CLASS_NAMES,
                "class_to_idx": {
                    name: index
                    for index, name in enumerate(CLASS_NAMES)
                },
                "image_size": 224,
                "best_val_accuracy": best_val_accuracy,
                "epoch": epoch + 1,
            },
            BEST_MODEL_PATH,
        )

        print("  Saved new best checkpoint.")

# Save training metrics.
with open(HISTORY_PATH, "w", encoding="utf-8") as file:
    json.dump(history, file, indent=4)

print("\nTraining completed.")
print(f"Best validation accuracy: {best_val_accuracy:.4f}")
print("Best model:", BEST_MODEL_PATH)
print("Training history:", HISTORY_PATH)
