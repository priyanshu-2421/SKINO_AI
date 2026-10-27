from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix


# =========================================================
# 1. CONFIGURATION
# =========================================================

DATA_DIR = Path("datasets/combined_skin_type_v3")
MODEL_DIR = Path("models")
RESULTS_DIR = Path("training_results")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 15
SEED = 42

CLASSES = ["combination", "dry", "normal", "oily"]

MODEL_PATH = MODEL_DIR / "skin_type_mobilenetv2.keras"
CLASS_PATH = MODEL_DIR / "skin_type_classes.json"

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}

np.random.seed(SEED)
tf.random.set_seed(SEED)

print("=" * 60)
print("SKINO_AI - IMPROVED MODEL TRAINING")
print("=" * 60)

print("TensorFlow:", tf.__version__)
print("Dataset:", DATA_DIR.resolve())


# =========================================================
# 2. CHECK DATASET
# =========================================================

if not DATA_DIR.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_DIR.resolve()}"
    )


def count_images(folder):
    return sum(
        1
        for path in folder.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )


for split in ["train", "valid", "test"]:
    for class_name in CLASSES:

        folder = DATA_DIR / split / class_name

        if not folder.exists():
            raise FileNotFoundError(
                f"Missing folder: {folder}"
            )

        number = count_images(folder)

        if number == 0:
            raise ValueError(
                f"No images found in: {folder}"
            )

        print(
            f"{split:5s} | {class_name:12s} | "
            f"{number} images"
        )


# =========================================================
# 3. LOAD DATASETS
# =========================================================

print("\nLoading datasets...")

train_ds = keras.utils.image_dataset_from_directory(
    DATA_DIR / "train",
    class_names=CLASSES,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=True,
    seed=SEED
)

valid_ds = keras.utils.image_dataset_from_directory(
    DATA_DIR / "valid",
    class_names=CLASSES,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

test_ds = keras.utils.image_dataset_from_directory(
    DATA_DIR / "test",
    class_names=CLASSES,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)


# =========================================================
# 4. COMPUTE CLASS WEIGHTS
# =========================================================

print("\nCalculating class weights...")

class_counts = np.zeros(len(CLASSES), dtype=np.int64)

for _, labels in train_ds:
    label_values, counts = np.unique(
        labels.numpy(),
        return_counts=True
    )

    for label, count in zip(label_values, counts):
        class_counts[label] += count

if np.any(class_counts == 0):
    raise ValueError(
        f"At least one class has no training images: {class_counts}"
    )

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(len(CLASSES)),
    y=np.repeat(
        np.arange(len(CLASSES)),
        class_counts
    )
)

class_weight_dict = {
    index: float(weight)
    for index, weight in enumerate(class_weights)
}

print("\nTraining class counts:")

for index, class_name in enumerate(CLASSES):
    print(
        f"{class_name}: {class_counts[index]} images | "
        f"weight: {class_weights[index]:.3f}"
    )


# =========================================================
# 5. PERFORMANCE OPTIMIZATION
# =========================================================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
valid_ds = valid_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)


# =========================================================
# 6. DATA AUGMENTATION
# =========================================================

augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.025),
        layers.RandomZoom(0.05),
    ],
    name="data_augmentation"
)


# =========================================================
# 7. LOAD MOBILENETV2
# =========================================================

print("\nLoading pretrained MobileNetV2...")

base_model = keras.applications.MobileNetV2(
    input_shape=(*IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)

# Initially freeze pretrained layers
base_model.trainable = False


# =========================================================
# 8. BUILD MODEL
# =========================================================

inputs = keras.Input(
    shape=(*IMG_SIZE, 3),
    name="input_image"
)

x = augmentation(inputs)

# MobileNetV2 preprocessing: pixels [0,255] -> [-1,1]
x = layers.Rescaling(
    scale=1.0 / 127.5,
    offset=-1.0,
    name="mobilenet_preprocessing"
)(x)

x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.35)(x)

x = layers.Dense(
    128,
    activation="relu"
)(x)

x = layers.Dropout(0.25)(x)

outputs = layers.Dense(
    len(CLASSES),
    activation="softmax",
    name="skin_type_output"
)(x)

model = keras.Model(
    inputs=inputs,
    outputs=outputs,
    name="SKINO_AI_MobileNetV2"
)


# =========================================================
# 9. COMPILE MODEL
# =========================================================

model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=0.0005
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# =========================================================
# 10. CALLBACKS
# =========================================================

callbacks = [

    keras.callbacks.ModelCheckpoint(
        filepath=str(MODEL_PATH),
        monitor="val_loss",
        save_best_only=True,
        mode="min",
        verbose=1
    ),

    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True,
        verbose=1
    ),

    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    ),

    keras.callbacks.CSVLogger(
        str(RESULTS_DIR / "training_history.csv")
    )
]


# =========================================================
# 11. TRAIN MODEL
# =========================================================

print("\nStarting training...")

history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS,
    class_weight=class_weight_dict,
    callbacks=callbacks
)


# =========================================================
# 12. SAVE TRAINING HISTORY
# =========================================================

history_data = {
    key: [float(value) for value in values]
    for key, values in history.history.items()
}

with open(
    RESULTS_DIR / "training_history.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        history_data,
        file,
        indent=4
    )

print("\nTraining history saved.")


# =========================================================
# 13. PLOT ACCURACY
# =========================================================

plt.figure(figsize=(9, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("SKINO_AI - Training vs Validation Accuracy")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "accuracy_graph.png",
    dpi=150
)

plt.close()


# =========================================================
# 14. PLOT LOSS
# =========================================================

plt.figure(figsize=(9, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("SKINO_AI - Training vs Validation Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "loss_graph.png",
    dpi=150
)

plt.close()


# =========================================================
# 15. LOAD BEST MODEL
# =========================================================

print("\nLoading best saved model...")

best_model = keras.models.load_model(
    MODEL_PATH
)

with open(
    CLASS_PATH,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        {
            "classes": CLASSES,
            "image_size": list(IMG_SIZE)
        },
        file,
        indent=4
    )


# =========================================================
# 16. FINAL TEST EVALUATION
# =========================================================

print("\nEvaluating test dataset...")

test_loss, test_accuracy = best_model.evaluate(
    test_ds,
    verbose=1
)

y_true = np.concatenate(
    [labels.numpy() for _, labels in test_ds]
)

probabilities = best_model.predict(
    test_ds,
    verbose=1
)

y_pred = np.argmax(
    probabilities,
    axis=1
)

print("\nCLASSIFICATION REPORT")

report = classification_report(
    y_true,
    y_pred,
    labels=list(range(len(CLASSES))),
    target_names=CLASSES,
    zero_division=0,
    digits=4
)

print(report)

print("\nCONFUSION MATRIX")

matrix = confusion_matrix(
    y_true,
    y_pred,
    labels=list(range(len(CLASSES)))
)

print("Class order:", CLASSES)
print(matrix)


# =========================================================
# 17. SAVE EVALUATION RESULTS
# =========================================================

with open(
    RESULTS_DIR / "evaluation_report.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write("SKINO_AI MODEL EVALUATION\n\n")
    file.write(f"Test Loss: {test_loss:.4f}\n")
    file.write(f"Test Accuracy: {test_accuracy * 100:.2f}%\n\n")
    file.write("Class order:\n")
    file.write(", ".join(CLASSES))
    file.write("\n\nClassification Report:\n")
    file.write(report)
    file.write("\nConfusion Matrix:\n")
    file.write(str(matrix))


# =========================================================
# 18. FINAL SUMMARY
# =========================================================

print("\n" + "=" * 60)
print("SKINO_AI TRAINING COMPLETED")
print("=" * 60)

print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

print("\nSaved model:")
print(MODEL_PATH.resolve())

print("\nSaved results:")
print(RESULTS_DIR.resolve())

print("\nGenerated files:")
print("- training_history.csv")
print("- training_history.json")
print("- accuracy_graph.png")
print("- loss_graph.png")
print("- evaluation_report.txt")