import os
import torch
import onnx
import onnxruntime as ort
import cv2
import numpy as np

print("=" * 60)
print("Python Environment Check")
print("=" * 60)

print("Conda env:", os.environ.get("CONDA_DEFAULT_ENV"))
print("torch:", torch.__version__)
print("torch cuda version:", torch.version.cuda)
print("torch cuda available:", torch.cuda.is_available())

if torch.cuda.is_available():
        print("gpu name:", torch.cuda.get_device_name(0))
        print("gpu count:", torch.cuda.device_count())

print("onnx:", onnx.__version__)
print("onnxruntime:", ort.__version__)
print("onnxruntime providers:", ort.get_available_providers())
print("opencv:", cv2.__version__)
print("numpy:", np.__version__)

print("=" * 60)
