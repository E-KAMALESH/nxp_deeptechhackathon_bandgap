import os
import cv2
import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from torchvision import transforms
import tkinter as tk
from tkinter import filedialog

# =========================
# CONFIG
# =========================
ONNX_PATH = "nxp_defect_model.onnx"

CLASSES = [
    'bridge',
    'clean',
    'cmp',
    'cracks',
    'ler',
    'opens',
    'other',
    'vias'
]

# =========================
# FILE PICKER (MULTI-SELECT)
# =========================
root = tk.Tk()
root.withdraw()  # hide main window

file_paths = filedialog.askopenfilenames(
    title="Select SEM images",
    filetypes=[("Image files", "*.png *.jpg *.jpeg")]
)

if not file_paths:
    print("❌ No images selected")
    exit()

print(f"📂 Selected {len(file_paths)} images\n")

# =========================
# PREPROCESS (same as training)
# =========================
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# =========================
# LOAD ONNX MODEL
# =========================
ort_session = ort.InferenceSession(ONNX_PATH)
input_name = ort_session.get_inputs()[0].name

# =========================
# RUN INFERENCE
# =========================
for img_path in file_paths:
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"❌ Failed to read {img_path}")
        continue

    img_tensor = transform(img).unsqueeze(0)  # (1,1,128,128)

    onnx_input = {input_name: img_tensor.numpy()}
    logits = ort_session.run(None, onnx_input)[0]
    pred_idx = int(np.argmax(logits))
    pred_class = CLASSES[pred_idx]

    print(f"{os.path.basename(img_path):30s} → 🧠 {pred_class}")
