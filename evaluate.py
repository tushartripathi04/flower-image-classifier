
from pathlib import Path
import json

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from src.dataset import create_dataloaders
from src.model import create_model, CLASS_NAMES


ROOT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ROOT_DIR / "models" / "efficientnet_b0_best.pth"
REPORT_DIR = ROOT_DIR / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model checkpoint not found: {MODEL_PATH}. "
        "Run python train.py first."
    )

# Load the saved model checkpoint.
checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True,
)

model = create_model(
    num_classes=len(CLASS_NAMES),
    freeze_features=True,
).to(device)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Obtain the untouched test loader.
_, _, test_loader = create_dataloaders(ROOT_DIR)

y_true = []
y_pred = []

with torch.inference_mode():
    for images, labels in test_loader:
        images = images.to(device)

        logits = model(images)
        predictions = logits.argmax(dim=1).cpu()

        y_true.extend(labels.tolist())
        y_pred.extend(predictions.tolist())

# Calculate evaluation metrics.
accuracy = accuracy_score(y_true, y_pred)

report = classification_report(
    y_true,
    y_pred,
    labels=list(range(len(CLASS_NAMES))),
    target_names=CLASS_NAMES,
    output_dict=True,
    zero_division=0,
)

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=list(range(len(CLASS_NAMES))),
)

print(f"Test images: {len(y_true)}")
print(f"Test accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Macro precision: {report['macro avg']['precision']:.4f}")
print(f"Macro recall: {report['macro avg']['recall']:.4f}")
print(f"Macro F1-score: {report['macro avg']['f1-score']:.4f}")

print("\nClassification report:")
print(
    classification_report(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        zero_division=0,
    )
)

print("Confusion matrix:")
print(cm)

# Save numerical results.
results = {
    "test_accuracy": accuracy,
    "macro_precision": report["macro avg"]["precision"],
    "macro_recall": report["macro avg"]["recall"],
    "macro_f1": report["macro avg"]["f1-score"],
    "classification_report": report,
    "confusion_matrix": cm.tolist(),
    "class_names": CLASS_NAMES,
}

with open(
    REPORT_DIR / "test_metrics.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(results, file, indent=4)

# Plot and save the confusion matrix.
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=CLASS_NAMES,
)

display.plot(
    cmap="Blues",
    xticks_rotation=30,
    values_format="d",
)

plt.title("EfficientNet-B0: Test Confusion Matrix")
plt.tight_layout()
plt.savefig(
    REPORT_DIR / "confusion_matrix.png",
    dpi=150,
    bbox_inches="tight",
)
plt.show()

print("\nEvaluation complete.")
print("Metrics saved to:", REPORT_DIR / "test_metrics.json")
print("Confusion matrix saved to:", REPORT_DIR / "confusion_matrix.png")
