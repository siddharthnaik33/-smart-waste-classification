
import tensorflow as tf
import numpy as np
import io

from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image


# =========================
# FastAPI Application
# =========================

app = FastAPI(
    title="Smart Waste Classification API",
    description="AI-based waste classification using MobileNetV2",
    version="1.0"
)


# =========================
# CORS Configuration
# =========================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# Model Path
# =========================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "best_finetuned_mobilenet.keras"


# =========================
# Load Model
# =========================

model = tf.keras.models.load_model(
    MODEL_PATH
)


# =========================
# Class Names
# =========================

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


IMAGE_SIZE = (160, 160)


# =========================
# Root Endpoint
# =========================

@app.get("/")
def home():

    return {
        "message": "Smart Waste Classification API",
        "status": "running"
    }


# =========================
# Prediction Endpoint
# =========================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    # Check uploaded file
    if not file.content_type or not file.content_type.startswith("image/"):

        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image file."
        )


    # Read uploaded image
    image_bytes = await file.read()


    # Open uploaded image
    try:

        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

    except Exception:

        raise HTTPException(
            status_code=400,
            detail="Unable to read the uploaded image."
        )


    # Resize image
    image = image.resize(
        IMAGE_SIZE
    )


    # Convert image to NumPy array
    image_array = np.array(
        image,
        dtype=np.float32
    )


    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )


    # Make prediction
    predictions = model.predict(
        image_array,
        verbose=0
    )[0]


    # Get highest probability
    predicted_index = np.argmax(
        predictions
    )


    predicted_class = CLASS_NAMES[
        predicted_index
    ]


    confidence = float(
        predictions[predicted_index]
    )


    # Return prediction
    return {
        "filename": file.filename,
        "predicted_class": predicted_class,
        "confidence": round(
            confidence * 100,
            2
        )
    }