from pathlib import Path
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix

# =====================================================
# 1. CONFIGURATION
# =====================================================

DATA_DIR = Path("datasets/combined_skin_type_v3")
MODEL_PATH = Path("models/skin_type_mobilenetv2.keras")
CLASS_PATH = Path("models/skin_type_classes.json")

IMG_SIZE = (224, 224)
BATCH_SIZE = 16

CLASSES = ["combination", "dry", "normal", "oily"]

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"
}

print("=" * 60)
print("SKINO_AI - MODEL EVALUATION")
print("=" * 60)


# =====================================================
# 2. CHECK PATHS
# =====================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Trained model not found: {MODEL_PATH.resolve()}"
    )

TEST_DIR = DATA_DIR / "test"

if not TEST_DIR.exists():
    raise FileNotFoundError(
        f"Test directory not found: {TEST_DIR.resolve()}"
    )


# =====================================================
# 3. COUNT TEST IMAGES
# =====================================================

print("\nChecking test images...")

total_images = 0

for class_name in CLASSES:
    class_dir = TEST_DIR / class_name

    if not class_dir.exists():
        raise FileNotFoundError(
            f"Class folder not found: {class_dir}"
        )

    count = sum(
        1
        for file in class_dir.rglob("*")
        if file.is_file()
        and file.suffix.lower() in IMAGE_EXTENSIONS
    )

    total_images += count

    print(f"{class_name}: {count} images")

print(f"\nTotal test images: {total_images}")

if total_images == 0:
    raise ValueError("No test images found!")


# =====================================================
# 4. LOAD TEST DATASET
# =====================================================

print("\nLoading test dataset...")

test_ds = keras.utils.image_dataset_from_directory(
    TEST_DIR,
    class_names=CLASSES,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)


# =====================================================
# 5. LOAD TRAINED MODEL
# =====================================================

print("\nLoading trained model...")

model = keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")

print("Expected input shape:", model.input_shape)
print("Number of output classes:", model.output_shape[-1])


# =====================================================
# 6. CHECK CLASS MAPPING
# =====================================================

if model.output_shape[-1] != len(CLASSES):
    raise ValueError(
        "Model output count does not match the four classes."
    )

if CLASS_PATH.exists():
    with open(CLASS_PATH, "r", encoding="utf-8") as file:
        saved_classes = json.load(file).get("classes", [])

    if saved_classes != CLASSES:
        raise ValueError(
            f"Class mapping mismatch!\n"
            f"Saved: {saved_classes}\n"
            f"Expected: {CLASSES}"
        )


# =====================================================
# 7. GENERATE PREDICTIONS
# =====================================================

print("\nGenerating predictions...")

y_true = np.concatenate([
    labels.numpy()
    for _, labels in test_ds
])

probabilities = model.predict(
    test_ds,
    verbose=1
)

y_pred = np.argmax(probabilities, axis=1)

print("\nPredictions completed!")


# =====================================================
# 8. CLASSIFICATION REPORT
# =====================================================

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_true,
        y_pred,
        labels=list(range(len(CLASSES))),
        target_names=CLASSES,
        zero_division=0,
        digits=4
    )
)


# =====================================================
# 9. CONFUSION MATRIX
# =====================================================

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

matrix = confusion_matrix(
    y_true,
    y_pred,
    labels=list(range(len(CLASSES)))
)

print("Class order:", CLASSES)
print(matrix)


# =====================================================
# 10. FINAL EVALUATION
# =====================================================

print("\n" + "=" * 60)
print("FINAL MODEL EVALUATION")
print("=" * 60)

test_loss, test_accuracy = model.evaluate(
    test_ds,
    verbose=1
)

print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

print("\nEvaluation completed successfully!")