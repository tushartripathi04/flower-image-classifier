
from pathlib import Path
import tarfile
import urllib.request

# Project paths
DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

DATASET_URL = (
    "https://storage.googleapis.com/"
    "download.tensorflow.org/example_images/flower_photos.tgz"
)

ARCHIVE_PATH = DATA_DIR / "flower_photos.tgz"
EXTRACT_DIR = DATA_DIR / "flower_photos"

if EXTRACT_DIR.exists():
    print("Dataset folder already exists:", EXTRACT_DIR)
else:
    # Download the dataset
    print("Downloading flower dataset...")

    urllib.request.urlretrieve(
        DATASET_URL,
        ARCHIVE_PATH
    )

    print("Download complete.")

    # Extract safely into the expected folder
    print("Extracting dataset...")

    with tarfile.open(ARCHIVE_PATH, "r:gz") as archive:
        base = DATA_DIR.resolve()

        for member in archive.getmembers():
            target = (DATA_DIR / member.name).resolve()

            if not target.is_relative_to(base):
                raise ValueError(
                    f"Unsafe archive path: {member.name}"
                )

        archive.extractall(DATA_DIR, filter="data")

    print("Extraction complete.")

print("Dataset location:", EXTRACT_DIR.resolve())
print("Dataset ready!")
