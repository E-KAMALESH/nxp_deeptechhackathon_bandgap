import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# 👇 IMPORT THE REAL MODEL
from train import DefectNet

# =====================
# CONFIG
# =====================
DATASET_DIR = "dataset/test"
MODEL_PATH = "defect_model.pth"
IMG_SIZE = 128
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================
# DATASET
# =====================
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

dataset = datasets.ImageFolder(DATASET_DIR, transform=transform)
class_names = dataset.classes
print("Classes:", class_names)

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# =====================
# LOAD MODEL (EXACT MATCH)
# =====================
model = DefectNet(num_classes=len(class_names)).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# =====================
# EVALUATION
# =====================
y_true, y_pred = [], []

with torch.no_grad():
    for images, labels in loader:
        images = images.to(DEVICE)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)

        y_true.extend(labels.numpy())
        y_pred.extend(preds.cpu().numpy())

# =====================
# METRICS
# =====================
acc = accuracy_score(y_true, y_pred)
print(f"\nAccuracy: {acc*100:.2f}%\n")

print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# =====================
# CONFUSION MATRIX
# =====================
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

print("✅ Confusion matrix saved as confusion_matrix.png")
