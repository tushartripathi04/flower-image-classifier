
import torch

from src.dataset import create_dataloaders

train_loader, val_loader, test_loader = create_dataloaders()

print("Train batches:", len(train_loader))
print("Validation batches:", len(val_loader))
print("Test batches:", len(test_loader))

# Fetch one batch of training images.
images, labels = next(iter(train_loader))

print("\nImage tensor shape:", images.shape)
print("Label tensor shape:", labels.shape)
print("Image data type:", images.dtype)
print("Label data type:", labels.dtype)
print("Labels in this batch:", labels.tolist())

assert images.ndim == 4
assert images.shape[1:] == (3, 224, 224)
assert labels.ndim == 1
assert images.shape[0] == labels.shape[0]
assert labels.min().item() >= 0
assert labels.max().item() <= 4

print("\nDataLoader verification successful!")
