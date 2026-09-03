import shutil
import random
from pathlib import Path

SOURCE_DIR = Path("data/RealWaste/realwaste-main/RealWaste")

TRAIN_DIR = Path("data/train")
VAL_DIR = Path("data/validation")
TEST_DIR = Path("data/test")

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

random.seed(42)

for class_folder in SOURCE_DIR.iterdir():

    if not class_folder.is_dir():
        continue

    class_name = class_folder.name

    images = [
        file for file in class_folder.iterdir()
        if file.suffix.lower() in IMAGE_EXTENSIONS
    ]

    random.shuffle(images)

    total = len(images)

    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]

    for output_dir in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        (output_dir / class_name).mkdir(
            parents=True,
            exist_ok=True
        )

    for image in train_images:
        shutil.copy2(image, TRAIN_DIR / class_name / image.name)

    for image in val_images:
        shutil.copy2(image, VAL_DIR / class_name / image.name)

    for image in test_images:
        shutil.copy2(image, TEST_DIR / class_name / image.name)

    print(
        f"{class_name}: "
        f"Train={len(train_images)}, "
        f"Validation={len(val_images)}, "
        f"Test={len(test_images)}"
    )

print("\nDataset split completed successfully!")