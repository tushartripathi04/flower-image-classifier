
import torch
import torch.nn as nn
from torchvision.models import (
    efficientnet_b0,
    EfficientNet_B0_Weights,
)


NUM_CLASSES = 5

CLASS_NAMES = [
    "daisy",
    "dandelion",
    "roses",
    "sunflowers",
    "tulips",
]


def create_model(
    num_classes: int = NUM_CLASSES,
    freeze_features: bool = True,
) -> nn.Module:
    """
    Load pretrained EfficientNet-B0 and replace its
    classification head for our flower classes.
    """

    weights = EfficientNet_B0_Weights.DEFAULT

    model = efficientnet_b0(weights=weights)

    # Freeze the pretrained feature extractor.
    if freeze_features:
        for parameter in model.features.parameters():
            parameter.requires_grad = False

    # EfficientNet-B0's final classifier is a Sequential module.
    in_features = model.classifier[1].in_features

    model.classifier[1] = nn.Linear(
        in_features,
        num_classes,
    )

    return model


if __name__ == "__main__":
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = create_model().to(device)
    model.eval()

    # Test with a dummy batch.
    dummy_images = torch.randn(
        2, 3, 224, 224, device=device
    )

    with torch.no_grad():
        logits = model(dummy_images)

    print("Device:", device)
    print("Model:", model.__class__.__name__)
    print("Input shape:", dummy_images.shape)
    print("Output shape:", logits.shape)

    trainable_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    total_parameters = sum(
        p.numel() for p in model.parameters()
    )

    print("Trainable parameters:", trainable_parameters)
    print("Total parameters:", total_parameters)

    assert logits.shape == (2, NUM_CLASSES)

    print("Model verification successful!")
