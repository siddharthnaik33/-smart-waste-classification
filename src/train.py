import tensorflow as tf
from pathlib import Path

# =========================
# Dataset paths
# =========================

TRAIN_DIR = Path("data/train")
VAL_DIR = Path("data/validation")
TEST_DIR = Path("data/test")

# =========================
# Settings
# =========================

IMAGE_SIZE = (160, 160)
BATCH_SIZE = 16
SEED = 42
EPOCHS = 10

# =========================
# Load datasets
# =========================

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# Get class names
class_names = train_ds.class_names
NUM_CLASSES = len(class_names)

print("\nClasses:")
print(class_names)

# =========================
# Data augmentation
# =========================

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])

# =========================
# Improve data performance
# =========================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

# =========================
# Build CNN model
# =========================

model = tf.keras.Sequential([
    
    tf.keras.layers.Input(shape=(160, 160, 3)),

    # Data augmentation
    data_augmentation,

    # Normalize pixel values
    tf.keras.layers.Rescaling(1./255),

    # CNN Block 1
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    # CNN Block 2
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    # CNN Block 3
    tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    # Classification layers
    tf.keras.layers.GlobalAveragePooling2D(),

    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(128, activation="relu"),

    tf.keras.layers.Dropout(0.3),

    # Output layer
    tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")
])

# =========================
# Compile model
# =========================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# =========================
# Display model summary
# =========================

print("\nModel Summary:\n")
model.summary()

print("\nModel built successfully!")
print("Number of classes:", NUM_CLASSES)

# =========================
# Training callbacks
# =========================

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "models/best_waste_classifier.keras",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=4,
    restore_best_weights=True,
    mode="max",
    verbose=1
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    min_lr=0.000001,
    verbose=1
)

# =========================
# Train model
# =========================

print("\nStarting model training...\n")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    callbacks=[
        checkpoint,
        early_stopping,
        reduce_lr
    ]
)

print("\nTraining completed successfully!")

# Save final model
model.save("models/final_waste_classifier.keras")

print("Final model saved successfully!")