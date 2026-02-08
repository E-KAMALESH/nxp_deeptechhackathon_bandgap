import os
import cv2
import numpy as np
import shutil

INPUT_DIR = "dataset/train"
OUTPUT_DIR = "dataset/train_processed"
IMG_SIZE = 128

# 🔒 Ensure clean output every run
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR)

print("Starting NXP-safe preprocessing...")

for cls in os.listdir(INPUT_DIR):
    in_dir = os.path.join(INPUT_DIR, cls)
    out_dir = os.path.join(OUTPUT_DIR, cls)

    if not os.path.isdir(in_dir):
        continue

    os.makedirs(out_dir, exist_ok=True)

    for img_name in os.listdir(in_dir):
        img_path = os.path.join(in_dir, img_name)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # Resize (ONLY geometric normalization)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE),
                         interpolation=cv2.INTER_AREA)

        # Normalize to [0,1]
        img = img.astype(np.float32) / 255.0

        # Save back as uint8 PNG
        img_to_save = (img * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(out_dir, img_name), img_to_save)

print("✅ NXP-safe preprocessing completed.")
