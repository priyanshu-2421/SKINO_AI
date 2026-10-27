import tensorflow as tf
from PIL import Image
import numpy as np
import os

MODEL_PATH = r"D:\SKINO_AI\models\skin_type_mobilenetv2.keras"
IMAGE_PATH = r"D:\SKINO_AI\test_images\WhatsApp Image 2026-10-09 at 3.30.27 PM.jpeg"

print("Loading saved model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")

class_names = ["Combination", "Dry", "Normal", "Oily"]

if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError(
        f"Photo not found: {IMAGE_PATH}\n"
        "Check the image filename and extension."
    )

image = Image.open(IMAGE_PATH).convert("RGB")
image = image.resize((160, 160))

image_array = np.array(image, dtype=np.float32)
image_array = np.expand_dims(image_array, axis=0)

predictions = model.predict(image_array, verbose=0)

predicted_index = np.argmax(predictions[0])
predicted_class = class_names[predicted_index]
confidence = predictions[0][predicted_index] * 100

print("\n🌸 SKINO_AI Skin Analysis")
print("--------------------------")
print("Estimated Skin Type:", predicted_class)
print(f"Model Confidence: {confidence:.2f}%")
print("\nNote: Experimental prediction, not a medical diagnosis.")