from pathlib import Path

import numpy as np
import tensorflow as tf

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from ai.graph import graph


# =========================================================
# Project paths
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "best_finetuned_mobilenet.keras"

UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)


# =========================================================
# FastAPI Application
# =========================================================

app = FastAPI(
    title="Smart Waste Classification API",
    description="AI-powered waste classification and recycling assistant",
    version="1.0.0"
)


# =========================================================
# CORS Configuration
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# Model settings
# =========================================================

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


# =========================================================
# Load MobileNetV2 model
# =========================================================

print("Loading MobileNetV2 model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully!")


# =========================================================
# Prediction function
# =========================================================

def predict_image(image_path: Path):

    image = tf.keras.utils.load_img(
        image_path,
        target_size=IMAGE_SIZE
    )

    image_array = tf.keras.utils.img_to_array(
        image
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    predictions = model.predict(
        image_array,
        verbose=0
    )

    predicted_index = np.argmax(
        predictions[0]
    )

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    confidence = float(
        predictions[0][predicted_index] * 100
    )

    return predicted_class, confidence


# =========================================================
# Health check
# =========================================================

@app.get("/")
def home():

    return {
        "message": "Smart Waste Classification API is running",
        "model": "MobileNetV2",
        "classes": CLASS_NAMES
    }


# =========================================================
# Image prediction API
# =========================================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    # -----------------------------------------------------
    # Check uploaded file
    # -----------------------------------------------------

    if not file.content_type or not file.content_type.startswith("image/"):
        return {
            "error": "Please upload an image file."
        }


    # -----------------------------------------------------
    # Save uploaded image
    # -----------------------------------------------------

    file_path = UPLOAD_DIR / file.filename

    contents = await file.read()

    with open(file_path, "wb") as f:
        f.write(contents)


    # -----------------------------------------------------
    # MobileNetV2 prediction
    # -----------------------------------------------------

    waste_class, confidence = predict_image(
        file_path
    )

    print(
        f"\n[Prediction] {waste_class} "
        f"({confidence:.2f}%)"
    )


    # -----------------------------------------------------
    # Run LangGraph AI Agent
    # -----------------------------------------------------

    print("[Agent] Running LangGraph...")

    result = graph.invoke({
        "waste_class": waste_class,
        "confidence": confidence,
        "context": "",
        "response": ""
    })


    # -----------------------------------------------------
    # Get AI-generated response
    # -----------------------------------------------------

    ai_response = result["response"]

    print("[Agent] Response generated successfully.")


    # -----------------------------------------------------
    # Return final API response
    # -----------------------------------------------------

    return {
        "filename": file.filename,
        "waste_class": waste_class,
        "confidence": round(confidence, 2),
        "message": (
            f"I identified this image as "
            f"{waste_class} with "
            f"{confidence:.2f}% confidence."
        ),
        "ai_response": ai_response
    }