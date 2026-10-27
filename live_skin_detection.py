import cv2
import tensorflow as tf
import numpy as np

# ==============================
# SKINO_AI - LIVE SKIN ANALYSIS
# ==============================

MODEL_PATH = r"D:\SKINO_AI\models\skin_type_mobilenetv2.keras"

CLASS_NAMES = [
    "Combination",
    "Dry",
    "Normal",
    "Oily"
]

print("Loading SKINO_AI model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")

# Load OpenCV face detector
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

# Start webcam
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
    camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError(
        "Webcam open nahi hua. Camera permission/device check karo."
    )

print("Webcam started!")
print("Press Q to quit.")

while True:

    success, frame = camera.read()

    if not success:
        print("Camera frame read nahi ho paya.")
        break

    # Mirror the frame
    frame = cv2.flip(frame, 1)

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(100, 100)
    )

    for (x, y, w, h) in faces:

        # Extract detected face
        face = frame[y:y+h, x:x+w]

        if face.size == 0:
            continue

        # OpenCV BGR -> RGB
        face_rgb = cv2.cvtColor(
            face,
            cv2.COLOR_BGR2RGB
        )

        # Resize to model input size
        face_rgb = cv2.resize(
            face_rgb,
            (160, 160)
        )

        # IMPORTANT:
        # Match preprocessing used during training.
        face_array = np.asarray(
            face_rgb,
            dtype=np.float32
        )

        face_array = np.expand_dims(
            face_array,
            axis=0
        )

        # Predict
        predictions = model.predict(
            face_array,
            verbose=0
        )[0]

        index = int(np.argmax(predictions))

        skin_type = CLASS_NAMES[index]

        confidence = float(
            predictions[index] * 100
        )

        # Draw face rectangle
        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            (80, 200, 120),
            2
        )

        # Display experimental prediction
        label = f"{skin_type}: {confidence:.1f}%"

        cv2.putText(
            frame,
            label,
            (x, max(y-10, 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (80, 200, 120),
            2
        )

    cv2.putText(
        frame,
        "SKINO_AI | Press Q to Quit",
        (15, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imshow(
        "SKINO_AI - Live Skin Analysis",
        frame
    )

    # Quit when Q is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()

print("SKINO_AI webcam closed.")