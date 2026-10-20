
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision.datasets import ImageFolder

DATA_DIR = Path("data/raw/flower_photos")
OUTPUT_DIR = Path("reports/splits")

RANDOM_STATE = 42
TEST_SIZE = 0.15
VALIDATION_SIZE = 0.15

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not DATA_DIR.exists():
    raise FileNotFoundError(
        "Dataset not found. Run python download_data.py first."
    )

dataset = ImageFolder(DATA_DIR)
records = []
bad_images = []

print("Checking image integrity...")

for image_path, label in dataset.samples:
    try:
        with Image.open(image_path) as image:
            image.verify()

        records.append({
            "path": image_path,
            "label": label,
            "class_name": dataset.classes[label],
        })

    except Exception:
        bad_images.append(str(image_path))

print("Unreadable images:", len(bad_images))

if bad_images:
    print("First unreadable paths:")
    for path in bad_images[:10]:
        print(path)

if not records:
    raise RuntimeError("No readable images found.")

df = pd.DataFrame(records)

# Split off the test set first.
train_val_df, test_df = train_test_split(
    df,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=df["label"],
)

# Validation should be 15% of the original dataset.
validation_fraction_of_remaining = (
    VALIDATION_SIZE / (1 - TEST_SIZE)
)

train_df, val_df = train_test_split(
    train_val_df,
    test_size=validation_fraction_of_remaining,
    random_state=RANDOM_STATE,
    stratify=train_val_df["label"],
)

# Save paths relative to the project root.
for split_name, split_df in [
    ("train", train_df),
    ("validation", val_df),
    ("test", test_df),
]:
    split_df = split_df.copy()

    split_df["path"] = split_df["path"].map(
        lambda path: Path(path).resolve().relative_to(
            Path.cwd().resolve()
        ).as_posix()
    )

    split_df.to_csv(
        OUTPUT_DIR / f"{split_name}.csv",
        index=False,
    )

    print(f"\n{split_name.title()} images: {len(split_df)}")
    print(
        split_df["class_name"]
        .value_counts()
        .reindex(dataset.classes, fill_value=0)
        .to_string()
    )

print("\nClass mapping:", dataset.class_to_idx)
print("\nSplit files saved in:", OUTPUT_DIR.resolve())
print("Dataset splitting completed successfully.")
