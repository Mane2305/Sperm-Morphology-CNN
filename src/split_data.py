import os
import shutil
import random

# Paths
RAW_DIR = "data/raw"
TRAIN_DIR = "data/train"
VAL_DIR = "data/val"
TEST_DIR = "data/test"

# Split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

random.seed(42)

classes = ["normal", "abnormal", "non_sperm"]

for cls in classes:
    cls_path = os.path.join(RAW_DIR, cls)
    images = os.listdir(cls_path)
    random.shuffle(images)

    total = len(images)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]

    for img in train_images:
        shutil.copy(
            os.path.join(cls_path, img),
            os.path.join(TRAIN_DIR, cls, img)
        )

    for img in val_images:
        shutil.copy(
            os.path.join(cls_path, img),
            os.path.join(VAL_DIR, cls, img)
        )

    for img in test_images:
        shutil.copy(
            os.path.join(cls_path, img),
            os.path.join(TEST_DIR, cls, img)
        )

    print(f"{cls}: {len(train_images)} train, {len(val_images)} val, {len(test_images)} test")

print("✅ Data split completed.")
