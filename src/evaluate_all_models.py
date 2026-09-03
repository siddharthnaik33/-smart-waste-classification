
import tensorflow as tf
from pathlib import Path

# ==========================================
# SETTINGS
# ==========================================

TEST_DIR = Path("data/test")
IMAGE_SIZE = (160, 160)
BATCH_SIZE = 16

MODEL_DIR = Path("models")

MODELS = [
    "best_mobilenet_waste.keras",
    "final_mobilenet_waste.keras",
    "best_finetuned_mobilenet.keras",
    "final_finetuned_mobilenet.keras",
]

# ==========================================
# LOAD TEST DATASET
# ==========================================

print("\nLoading test dataset...\n")

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("\nClasses:")
print(test_ds.class_names)

test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

# ==========================================
# EVALUATE ALL MODELS
# ==========================================

results = []

for model_name in MODELS:

    model_path = MODEL_DIR / model_name

    print("\n" + "=" * 60)
    print(f"Evaluating: {model_name}")
    print("=" * 60)

    if not model_path.exists():
        print(f"Model not found: {model_path}")
        continue

    try:
        model = tf.keras.models.load_model(model_path)

        loss, accuracy = model.evaluate(
            test_ds,
            verbose=1
        )

        results.append({
            "model": model_name,
            "loss": loss,
            "accuracy": accuracy * 100
        })

        print(f"\nModel: {model_name}")
        print(f"Test Loss: {loss:.4f}")
        print(f"Test Accuracy: {accuracy * 100:.2f}%")

        del model

    except Exception as e:
        print(f"ERROR loading/evaluating {model_name}")
        print(e)

# ==========================================
# FINAL COMPARISON
# ==========================================

print("\n\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

if results:

    results.sort(
        key=lambda x: x["accuracy"],
        reverse=True
    )

    for i, result in enumerate(results, start=1):

        print(
            f"{i}. {result['model']:<40} "
            f"Accuracy: {result['accuracy']:.2f}% "
            f"Loss: {result['loss']:.4f}"
        )

    best = results[0]

    print("\n" + "=" * 70)
    print("BEST MODEL")
    print("=" * 70)

    print(f"Model: {best['model']}")
    print(f"Test Accuracy: {best['accuracy']:.2f}%")
    print(f"Test Loss: {best['loss']:.4f}")

else:
    print("No models were successfully evaluated.")
