
from pathlib import Path
from collections import Counter

import matplotlib.pyplot as plt
from PIL import Image
from torchvision.datasets import ImageFolder

DATA_DIR = Path("data/raw/flower_photos")

if not DATA_DIR.exists():
    raise FileNotFoundError(
        "Dataset not found. Run python download_data.py first."
    )

# Read class labels and image paths
dataset = ImageFolder(DATA_DIR)

print("Number of images:", len(dataset))
print("Classes:", dataset.classes)
print("Class-to-index mapping:", dataset.class_to_idx)

# Count images in each class
counts = Counter(
    dataset.classes[label]
    for _, label in dataset.samples
)

print("\nImages per class:")
for class_name in dataset.classes:
    print(f"{class_name}: {counts[class_name]}")

# Check for unreadable images
bad_images = []

for image_path, _ in dataset.samples:
    try:
        with Image.open(image_path) as img:
            img.verify()
    except Exception:
        bad_images.append(image_path)

print("\nUnreadable images:", len(bad_images))

# Plot class distribution
plt.figure(figsize=(9, 5))
plt.bar(counts.keys(), counts.values())
plt.title("Flower Dataset: Class Distribution")
plt.xlabel("Flower class")
plt.ylabel("Number of images")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

# Display example images
fig, axes = plt.subplots(1, 5, figsize=(15, 4))

for class_index, class_name in enumerate(dataset.classes):
    image_path, _ = next(
        (p, label)
        for p, label in dataset.samples
        if label == class_index
    )

    with Image.open(image_path) as img:
        axes[class_index].imshow(img.convert("RGB"))

    axes[class_index].set_title(class_name)
    axes[class_index].axis("off")

plt.tight_layout()
plt.show()
