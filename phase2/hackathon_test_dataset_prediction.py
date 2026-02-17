"""
PHASE 2 - FINAL SUBMISSION
File Name: hackathon_test_dataset_prediction.py
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
from datetime import datetime
import os
import pandas as pd

from train import DefectNet

# =====================
# CONFIGURATION
# =====================
DATASET_DIR = "dataset\\nxp_processed"
MODEL_PATH = "defect_model.pth"
IMG_SIZE = 128
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================
# PARAMETERS (UNCHANGED)
# =====================
OPTIMIZED_PARAMS = {
    "temperature": 1.8,
    "min_confidence": 0.55,
    "max_confidence_other": 0.45,
    "min_margin": 0.15,
    "max_entropy": 0.7,
    "other_bias": 1.2,
    "min_conditions_for_other": 2,
    "top2_sum_threshold": 0.75,
}

TRAIN_CLASS_ORDER = ['bridge', 'clean', 'cmp', 'cracks', 'ler', 'opens', 'other', 'vias']
OTHER_CLASS_IDX = 6

mapping = {
    'Bridge': 'bridge', 'Clean': 'clean', 'CMP': 'cmp', 'Crack': 'cracks',
    'LER': 'ler', 'Open': 'opens', 'Other': 'other', 'VIA': 'vias'
}

# =====================
# OPTIMIZED PREDICTOR (UNCHANGED)
# =====================
class OptimizedPredictor:
    def __init__(self, params):
        self.params = params

    def predict(self, logits):
        scaled_logits = logits / self.params["temperature"]
        probs = torch.softmax(scaled_logits, dim=0)

        probs[OTHER_CLASS_IDX] *= self.params["other_bias"]
        probs = probs / (probs.sum() + 1e-10)

        top_probs, top_indices = torch.topk(probs, 2)

        confidence = top_probs[0].item()
        prediction = top_indices[0].item()
        second_confidence = top_probs[1].item()

        margin = confidence - second_confidence
        entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
        top2_sum = confidence + second_confidence

        conditions_met = 0
        if confidence < self.params["min_confidence"]:
            conditions_met += 1
        if margin < self.params["min_margin"]:
            conditions_met += 1
        if entropy > self.params["max_entropy"]:
            conditions_met += 1
        if top2_sum < self.params["top2_sum_threshold"]:
            conditions_met += 1

        if prediction == OTHER_CLASS_IDX and confidence > self.params["max_confidence_other"]:
            return OTHER_CLASS_IDX

        if conditions_met >= self.params["min_conditions_for_other"]:
            return OTHER_CLASS_IDX

        return prediction


# =====================
# MAIN EVALUATION
# =====================
def evaluate():

    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(DATASET_DIR, transform=transform)

    hack_idx_to_train_idx = {}
    for hack_name, hack_idx in dataset.class_to_idx.items():
        train_name = mapping[hack_name]
        train_idx = TRAIN_CLASS_ORDER.index(train_name)
        hack_idx_to_train_idx[hack_idx] = train_idx

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = DefectNet(num_classes=len(TRAIN_CLASS_ORDER)).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    predictor = OptimizedPredictor(OPTIMIZED_PARAMS)

    y_true = []
    y_pred = []
    prediction_log = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            outputs = model(images)

            for i in range(len(images)):
                pred = predictor.predict(outputs[i])
                y_pred.append(pred)
                prediction_log.append(pred)

            for label in labels.numpy():
                y_true.append(hack_idx_to_train_idx[label])

    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=TRAIN_CLASS_ORDER)

    print("\n===================================")
    print(f"FINAL ACCURACY: {accuracy*100:.2f}%")
    print("===================================")
    print("\nClassification Report:")
    print(report)

    # Save metrics
    with open("metrics.txt", "w") as f:
        f.write(f"Accuracy: {accuracy*100:.2f}%\n\n")
        f.write(report)

    # Save prediction log
    with open("prediction_log.txt", "w") as f:
        f.write(f"Prediction Run: {datetime.now()}\n\n")
        for i in range(len(y_pred)):
            f.write(f"True: {TRAIN_CLASS_ORDER[y_true[i]]} | Pred: {TRAIN_CLASS_ORDER[y_pred[i]]}\n")

    # Save CSV file
    df = pd.DataFrame({
        "True_Label": [TRAIN_CLASS_ORDER[i] for i in y_true],
        "Predicted_Label": [TRAIN_CLASS_ORDER[i] for i in y_pred]
    })
    df.to_csv("predictions.csv", index=False)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=TRAIN_CLASS_ORDER,
                yticklabels=TRAIN_CLASS_ORDER)
    plt.title(f'Confusion Matrix (Acc: {accuracy*100:.2f}%)')
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.close()

    print("\nFiles Generated:")
    print(" - metrics.txt")
    print(" - prediction_log.txt")
    print(" - predictions.csv")
    print(" - confusion_matrix.png")

    return accuracy


if __name__ == "__main__":
    evaluate()
