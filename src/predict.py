import tensorflow as tf
import numpy as np
from pathlib import Path

# =========================
# Settings
# =========================

MODEL_PATH = "models/best_finetuned_mobilenet.keras"

IMAGE_SIZE = (160, 160)

CLASS_NAMES = [
    "Cardboard",
    "Food Organics",
    "Glass",
    "Metal",
    "Miscellaneous Trash",
    "Paper",
    "Plastic",
    "Textile Trash",
    "Vegetation"
]


# =========================
# Load model
# =========================

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")


# =========================
# Prediction function
# =========================

def predict_image(image_path):

    image_path = Path(image_path)

    if not image_path.exists():
        print(f"Image not found: {image_path}")
        return

    # Load image
    image = tf.keras.utils.load_img(
        image_path,
        target_size=IMAGE_SIZE
    )

    # Convert to array
    image_array = tf.keras.utils.img_to_array(image)

    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # Predict
    predictions = model.predict(
        image_array,
        verbose=0
    )

    # Get predicted class
    predicted_index = np.argmax(predictions[0])

    predicted_class = CLASS_NAMES[predicted_index]

    confidence = predictions[0][predicted_index] * 100

    print("\n========== PREDICTION ==========")
    print(f"Image: {image_path.name}")
    print(f"Predicted Class: {predicted_class}")
    print(f"Confidence: {confidence:.2f}%")
    print("================================")


# =========================
# Main
# =========================

if __name__ == "__main__":

    image_path = input(
        "Enter image path: "
    )

    predict_image(image_path)