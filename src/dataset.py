
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# Match the preprocessing used for pretrained ImageNet models.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

IMAGE_SIZE = 224
BATCH_SIZE = 16


# Random transformations used only during training.
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# Deterministic transformations for validation and testing.
eval_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


class FlowerDataset(Dataset):
    def __init__(self, csv_file, root_dir=".", transform=None):
        self.root_dir = Path(root_dir).resolve()
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]

        image_path = self.root_dir / row["path"]

        with Image.open(image_path) as image:
            image = image.convert("RGB")

        label = int(row["label"])

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def create_dataloaders(root_dir="."):
    root_dir = Path(root_dir).resolve()
    split_dir = root_dir / "reports" / "splits"

    train_dataset = FlowerDataset(
        split_dir / "train.csv",
        root_dir=root_dir,
        transform=train_transform,
    )

    val_dataset = FlowerDataset(
        split_dir / "validation.csv",
        root_dir=root_dir,
        transform=eval_transform,
    )

    test_dataset = FlowerDataset(
        split_dir / "test.csv",
        root_dir=root_dir,
        transform=eval_transform,
    )

    # Keep num_workers=0 for straightforward Windows compatibility.
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, val_loader, test_loader
