
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from src.dataset import eval_transform
from src.model import CLASS_NAMES, create_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = PROJECT_ROOT / "models" / "efficientnet_b0_best.pth"


class FlowerPredictor:
    """Load the trained flower classifier and run image inference."""

    def __init__(
        self,
        checkpoint_path: Path = DEFAULT_CHECKPOINT,
        device: str | None = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)

        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")

        if not self.checkpoint_path.is_file():
            raise FileNotFoundError("Model checkpoint was not found.")

        # Load the checkpoint once when this predictor is initialized.
        checkpoint = torch.load(
            self.checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )

        if not isinstance(checkpoint, dict):
            raise ValueError("Invalid model checkpoint format.")

        class_names = checkpoint.get("class_names")

        if class_names != CLASS_NAMES:
            raise ValueError("Checkpoint class order does not match the source.")

        state_dict = checkpoint.get("model_state_dict")

        if not isinstance(state_dict, dict):
            raise ValueError("Checkpoint is missing model weights.")

        classifier_weight = state_dict.get("classifier.1.weight")

        if (
            not isinstance(classifier_weight, torch.Tensor)
            or tuple(classifier_weight.shape) != (len(CLASS_NAMES), 1280)
        ):
            raise ValueError("Checkpoint classifier shape is invalid.")

        # The checkpoint contains the trained parameters, so pretrained
        # ImageNet weights do not need to be downloaded again.
        self.model = create_model(
            num_classes=len(CLASS_NAMES),
            freeze_features=True,
            weights=None,
        )

        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device)
        self.model.eval()

        self.class_names = list(CLASS_NAMES)

    def predict(
        self,
        image: Image.Image,
        top_k: int = 3,
    ) -> dict:
        """Predict the flower class from a PIL image."""

        if not 1 <= top_k <= len(self.class_names):
            raise ValueError(
                f"top_k must be between 1 and {len(self.class_names)}."
            )

        # Match the RGB conversion used by the existing CLI.
        image = image.convert("RGB")
        image_tensor = eval_transform(image).unsqueeze(0)
        image_tensor = image_tensor.to(self.device)

        with torch.inference_mode():
            logits = self.model(image_tensor)
            scores = F.softmax(logits, dim=1)[0]
            top_scores, top_indices = torch.topk(scores, k=top_k)

        predictions = [
            {
                "label": self.class_names[index],
                "class_index": index,
                "score": float(score),
            }
            for score, index in zip(
                top_scores.tolist(),
                top_indices.tolist(),
            )
        ]

        return {
            "predicted_label": predictions[0]["label"],
            "class_index": predictions[0]["class_index"],
            "top_k": predictions,
            "score_type": "softmax_score",
        }
