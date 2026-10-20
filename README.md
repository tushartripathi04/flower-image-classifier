# Flower Image Classifier — Transfer Learning with EfficientNet-B0

A computer vision project that classifies flower photographs into five categories using transfer learning with a pretrained EfficientNet-B0 model.

## Overview

This project demonstrates a complete image-classification workflow, including dataset exploration, image preprocessing, data augmentation, model training, evaluation, and inference.

## Flower Categories

- Daisy
- Dandelion
- Roses
- Sunflowers
- Tulips

## Technology Stack

- Python
- PyTorch and Torchvision
- EfficientNet-B0
- NumPy and Pandas
- Matplotlib
- Scikit-learn
- Pillow

## Dataset

The project uses the TensorFlow Flowers dataset, containing 3,670 images across five flower categories.

Dataset reference: https://www.tensorflow.org/datasets/catalog/tf_flowers

The dataset is split into training, validation, and test sets using stratified sampling.

| Split | Images |
|---|---:|
| Training | 2,568 |
| Validation | 551 |
| Test | 551 |
| Total | 3,670 |

## Model Architecture

The project uses ImageNet-pretrained EfficientNet-B0.

The original classification head is replaced with a five-class linear layer. In the initial experiment, the pretrained feature extractor is frozen and only the new classification head is trained.

## Evaluation Results

Results from the initial three-epoch experiment:

| Metric | Result |
|---|---:|
| Best validation accuracy | 92.01% |
| Test accuracy | 91.83% |
| Macro precision | 91.88% |
| Macro recall | 91.74% |
| Macro F1-score | 91.77% |

These results were obtained from the held-out test set of 551 images. Performance may differ on photographs from other sources.

## Project Structure

```text
flower-image-classifier/
├── data/
│   └── raw/
├── notebooks/
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   └── model.py
├── models/
│   └── efficientnet_b0_best.pth
├── reports/
│   ├── splits/
│   ├── training_history.json
│   ├── test_metrics.json
│   └── confusion_matrix.png
├── download_data.py
├── explore_data.py
├── split_data.py
├── test_dataloader.py
├── train.py
├── evaluate.py
├── predict.py
├── requirements.txt
└── README.md
```

## Setup

Create and activate a virtual environment, then install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Download and prepare the dataset:

```powershell
python download_data.py
python explore_data.py
python split_data.py
python test_dataloader.py
```

## Train the Model

```powershell
python train.py
```

The best validation checkpoint is saved to:

```text
models/efficientnet_b0_best.pth
```

## Evaluate the Model

```powershell
python evaluate.py
```

The evaluation metrics and confusion matrix are saved in the `reports/` directory.

## Predict a Flower Image

```powershell
python predict.py "path/to/flower.jpg"
```

The script returns the predicted class, its softmax confidence score, and the top three predictions.

## Limitations

- The classifier recognizes only the five flower categories represented during training.
- Predictions may be unreliable for unfamiliar objects, poor-quality images, or substantially different image conditions.
- Softmax scores are not calibrated probabilities of correctness.
- The initial experiment freezes the pretrained feature extractor; further fine-tuning and calibration may improve performance.

## Future Improvements

- Fine-tune selected EfficientNet layers.
- Compare EfficientNet-B0 with other pretrained architectures.
- Analyze per-class errors and confidence calibration.
- Add a FastAPI inference endpoint and a web interface.
- Package the inference application using Docker.

## Author

Tushar Tripathi

Computer Science and Engineering
