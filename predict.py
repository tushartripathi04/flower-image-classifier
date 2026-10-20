
from pathlib import Path
import argparse

import torch
import torch.nn.functional as F
from PIL import Image

from src.dataset import eval_transform
from src.model import create_model, CLASS_NAMES


ROOT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ROOT_DIR / "models" / "efficientnet_b0_best.pth"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def load_trained_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {MODEL_PATH}"
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True,
    )

    class_names = checkpoint.get(
        "class_names", CLASS_NAMES
    )

    model = create_model(
        num_classes=len(class_names),
        freeze_features=True,
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model, class_names


def predict_image(image_path, model, class_names):
    image_path = Path(image_path)

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image_tensor = eval_transform(image).unsqueeze(0)

    image_tensor = image_tensor.to(device)

    with torch.inference_mode():
        logits = model(image_tensor)
        probabilities = F.softmax(logits, dim=1)[0]

    top_count = min(3, len(class_names))
    top_scores, top_indices = torch.topk(
        probabilities, k=top_count
    )

    predicted_index = top_indices[0].item()
    predicted_class = class_names[predicted_index]
    confidence = top_scores[0].item() * 100

    return predicted_class, confidence, [
        (class_names[index.item()], score.item() * 100)
        for score, index in zip(top_scores, top_indices)
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Predict the flower class of an image."
    )

    parser.add_argument(
        "image",
        help="Path to the flower image",
    )

    args = parser.parse_args()

    model, class_names = load_trained_model()

    predicted_class, confidence, top_predictions = (
        predict_image(args.image, model, class_names)
    )

    print("\nFlower Image Classifier")
    print("-" * 32)
    print("Predicted flower:", predicted_class)
    print(f"Confidence score: {confidence:.2f}%")

    print("\nTop predictions:")

    for flower, score in top_predictions:
        print(f"{flower:12s} {score:.2f}%")

    print(
        "\nNote: confidence scores are softmax outputs, "
        "not guaranteed probabilities of correctness."
    )


if __name__ == "__main__":
    main()
