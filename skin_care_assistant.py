import cv2
import numpy as np
import tensorflow as tf

from recommendation_engine import get_recommendations

# -----------------------------
# 1. Load trained skin model
# -----------------------------
MODEL_PATH = r"D:\SKINO_AI\models\skin_type_mobilenetv2.keras"

IMG_SIZE = (160, 160)

print("Loading SKINO_AI model...")

model = tf.keras.models.load_model(MODEL_PATH)

# Keep the same class order used during model training
CLASS_NAMES = ["Combination", "Dry", "Normal", "Oily"]

# -----------------------------
# 2. Initialize face detector
# -----------------------------
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

if face_detector.empty():
    raise RuntimeError("Face detector could not be loaded.")

# -----------------------------
# 3. Start webcam
# -----------------------------
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
    camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError(
        "Webcam could not be opened. Check camera permissions."
    )

print("Webcam started. Press Q to quit.")

last_skin_type = None
last_confidence = 0.0

try:
    while True:
        success, frame = camera.read()

        if not success:
            print("Could not read webcam frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80)
        )

        for (x, y, w, h) in faces:

            # Crop face
            face = frame[y:y + h, x:x + w]

            if face.size == 0:
                continue

            # Prepare image for model
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

            face_resized = cv2.resize(
                face_rgb,
                IMG_SIZE
            )

            image_array = face_resized.astype(np.float32)

            # Match MobileNetV2 preprocessing used during training
            image_array = image_array / 127.5 - 1.0

            image_array = np.expand_dims(
                image_array,
                axis=0
            )

            # Predict skin type
            prediction = model.predict(
                image_array,
                verbose=0
            )[0]

            predicted_index = int(np.argmax(prediction))

            last_skin_type = CLASS_NAMES[predicted_index]

            last_confidence = float(
                prediction[predicted_index] * 100
            )

            # Draw face rectangle
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            # Display prediction
            label = (
                f"{last_skin_type} "
                f"({last_confidence:.1f}%)"
            )

            cv2.putText(
                frame,
                label,
                (x, max(y - 10, 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        cv2.imshow(
            "SKINO_AI - Skin Care Assistant",
            frame
        )

        # Press Q to close webcam
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    camera.release()
    cv2.destroyAllWindows()

# -----------------------------
# 4. Show recommendations
# -----------------------------
if last_skin_type is not None:

    print("\n" + "=" * 45)
    print("       SKINO_AI SKIN CARE ASSISTANT")
    print("=" * 45)

    print(f"\nPredicted Skin Type: {last_skin_type}")
    print(f"Model Confidence Score: {last_confidence:.1f}%")

    recommendations = get_recommendations(last_skin_type)

    if recommendations:
        print("\n--- YOUR SKINCARE ROUTINE ---")

        for category, recommendation in recommendations.items():
            print(f"\n{category}:")
            print(recommendation)

    print("\nNote: This prediction is experimental.")
    print("It is not a medical skin diagnosis.")

else:
    print("No face was detected.")