import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

# =========================
# Settings
# =========================

TEST_DIR = Path("data/test")
MODEL_PATH = "models/best_mobilenet_waste.keras"

IMAGE_SIZE = (160, 160)
BATCH_SIZE = 16

# =========================
# Load dataset
# =========================

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = test_ds.class_names

# =========================
# Load model
# =========================

model = tf.keras.models.load_model(MODEL_PATH)

print("\nMobileNetV2 model loaded successfully!")
print("\nGenerating predictions...\n")

# =========================
# Predictions
# =========================

y_true = []
y_pred = []

for images, labels in test_ds:
    predictions = model.predict(images, verbose=0)

    predicted_classes = np.argmax(
        predictions,
        axis=1
    )

    y_true.extend(labels.numpy())
    y_pred.extend(predicted_classes)

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# =========================
# Classification Report
# =========================

print("\n========== CLASSIFICATION REPORT ==========\n")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4
    )
)

# =========================
# Confusion Matrix
# =========================

cm = confusion_matrix(y_true, y_pred)

print("\n========== CONFUSION MATRIX ==========\n")
print(cm)

# =========================
# Save confusion matrix
# =========================

fig, ax = plt.subplots(figsize=(12, 10))

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

display.plot(
    ax=ax,
    xticks_rotation=45
)

plt.title("MobileNetV2 Waste Classification Confusion Matrix")
plt.tight_layout()

plt.savefig(
    "models/mobilenet_confusion_matrix.png",
    dpi=300
)

plt.show()

print("\nConfusion matrix saved successfully!")
print("models/mobilenet_confusion_matrix.png")