import tensorflow as tf
from pathlib import Path

# =========================
# Settings
# =========================

TEST_DIR = Path("data/test")
MODEL_PATH = "models/best_finetuned_mobilenet.keras"

IMAGE_SIZE = (160, 160)
BATCH_SIZE = 16

# =========================
# Load test dataset
# =========================

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = test_ds.class_names

print("\nClasses:")
print(class_names)

# =========================
# Optimize dataset
# =========================

AUTOTUNE = tf.data.AUTOTUNE
test_ds = test_ds.prefetch(AUTOTUNE)

# =========================
# Load best fine-tuned model
# =========================

model = tf.keras.models.load_model(MODEL_PATH)

print("\nBest fine-tuned model loaded successfully!")

# =========================
# Evaluate
# =========================

print("\nEvaluating fine-tuned model...\n")

test_loss, test_accuracy = model.evaluate(test_ds)

print("\n========== FINE-TUNED MODEL RESULTS ==========")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print("===============================================")