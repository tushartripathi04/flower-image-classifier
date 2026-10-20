
import sys
import torch
import torchvision
import numpy as np
import pandas as pd
import sklearn
from PIL import Image

print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("NumPy:", np.__version__)
print("Pandas:", pd.__version__)
print("Scikit-learn:", sklearn.__version__)
print("CUDA available:", torch.cuda.is_available())

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

x = torch.rand(3, 224, 224).to(device)

print("Selected device:", device)
print("Test tensor shape:", x.shape)
print("Setup verification successful!")
