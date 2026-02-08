import os
import shutil
import random

SOURCE_DIR = "dataset/train_processed"
VAL_DIR = "dataset/val"
VAL_RATIO = 0.15   # 15% validation

random.seed(42)

# 🔒 Ensure clean val directory every run
if os.path.exists(VAL_DIR):
    shutil.rmtree(VAL_DIR)
os.makedirs(VAL_DIR)

for cls in os.listdir(SOURCE_DIR):
    src_cls_dir = os.path.join(SOURCE_DIR, cls)
    val_cls_dir = os.path.join(VAL_DIR, cls)

    if not os.path.isdir(src_cls_dir):
        continue

    os.makedirs(val_cls_dir, exist_ok=True)

    images = [
        f for f in os.listdir(src_cls_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    random.shuffle(images)
    num_val = int(len(images) * VAL_RATIO)
    val_images = images[:num_val]

    for img in val_images:
        shutil.move(
            os.path.join(src_cls_dir, img),
            os.path.join(val_cls_dir, img)
        )

    print(f"{cls}: {len(val_images)} images moved to val")
