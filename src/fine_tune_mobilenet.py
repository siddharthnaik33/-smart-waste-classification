import tensorflow as tf
from pathlib import Path

# =========================
# Settings
# =========================

TRAIN_DIR = Path("data/train")
VAL_DIR = Path("data/validation")

MODEL_PATH = "models/best_mobilenet_waste.keras"
FINE_TUNED_MODEL_PATH = "models/best_finetuned_mobilenet.keras"

IMAGE_SIZE = (160, 160)
BATCH_SIZE = 16
EPOCHS = 8

# =========================
# Load datasets
# =========================

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# =========================
# Load trained model
# =========================

model = tf.keras.models.load_model(MODEL_PATH)

print("\nModel loaded successfully!")

# Find MobileNetV2 base model
base_model = None

for layer in model.layers:
    if "mobilenetv2" in layer.name.lower():
        base_model = layer
        break

if base_model is None:
    raise ValueError("MobileNetV2 base model not found!")

print("Base model:", base_model.name)

# =========================
# Unfreeze upper layers
# =========================

base_model.trainable = True

# Freeze most layers
for layer in base_model.layers[:-30]:
    layer.trainable = False

print("\nFine-tuning last 30 MobileNetV2 layers.")

# =========================
# Recompile with low LR
# =========================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.00001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# =========================
# Callbacks
# =========================

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    FINE_TUNED_MODEL_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=3,
    restore_best_weights=True,
    mode="max",
    verbose=1
)

# =========================
# Fine-tune
# =========================

print("\nStarting fine-tuning...\n")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[
        checkpoint,
        early_stopping
    ]
)

print("\nFine-tuning completed successfully!")

model.save(
    "models/final_finetuned_mobilenet.keras"
)

print("Fine-tuned model saved successfully!")