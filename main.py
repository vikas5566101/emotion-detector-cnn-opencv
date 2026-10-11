from pathlib import Path

import cv2
import numpy as np
from tensorflow.keras.models import load_model


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.keras"


# OpenCV normally provides this cascade already.
CASCADE_PATH = (
    Path(cv2.data.haarcascades)
    / "haarcascade_frontalface_default.xml"
)


# --------------------------------------------------
# 2. Emotion labels
# IMPORTANT: Match your CNN's training class order.
# --------------------------------------------------

EMOTION_LABELS = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise",
]


# --------------------------------------------------
# 3. Load the trained CNN
# --------------------------------------------------

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

model = load_model(str(MODEL_PATH))

print("CNN loaded successfully!")
print("Input shape:", model.input_shape)
print("Output shape:", model.output_shape)


# --------------------------------------------------
# 4. Load Haar Cascade
# --------------------------------------------------

face_classifier = cv2.CascadeClassifier(
    str(CASCADE_PATH)
)

if face_classifier.empty():
    raise RuntimeError(
        "Could not load Haar Cascade."
    )

print("Face detector loaded successfully!")


# --------------------------------------------------
# 5. Start webcam
# --------------------------------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError(
        "Could not open webcam. Check camera access."
    )


# --------------------------------------------------
# 6. Real-time emotion detection
# --------------------------------------------------

try:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Could not read webcam frame.")
            break

        # Mirror the frame like a selfie camera.
        frame = cv2.flip(frame, 1)

        # Convert to grayscale.
        gray = cv2.cvtColor(
            frame, cv2.COLOR_BGR2GRAY
        )

        # Detect faces.
        faces = face_classifier.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(30, 30),
        )

        if len(faces) == 0:
            cv2.putText(
                frame,
                "No face detected",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

        for (x, y, w, h) in faces:

            # Crop the face.
            roi_gray = gray[y:y + h, x:x + w]

            # Resize to the assumed CNN input size.
            roi_gray = cv2.resize(
                roi_gray,
                (48, 48),
                interpolation=cv2.INTER_AREA,
            )

            # Normalize pixel values.
            roi = roi_gray.astype(
                np.float32
            )

            # Shape: (1, 48, 48, 1)
            roi = np.expand_dims(roi, axis=-1)
            roi = np.expand_dims(roi, axis=0)

            # Predict emotion.
            predictions = model.predict(
                roi, verbose=0
            )[0]

            emotion_index = int(
                np.argmax(predictions)
            )

            # Validate output class mapping.
            if emotion_index >= len(EMOTION_LABELS):
                raise ValueError(
                    "Model output count does not match "
                    "the configured emotion labels."
                )

            emotion = EMOTION_LABELS[emotion_index]
            confidence = float(
                predictions[emotion_index]
            )

            # Draw face rectangle.
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 255),
                2,
            )

            # Display emotion and score.
            label = (
                f"{emotion} ({confidence:.0%})"
            )

            text_y = max(y - 10, 25)

            cv2.putText(
                frame,
                label,
                (x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        # Display the webcam frame.
        cv2.imshow(
            "CNN Emotion Detector",
            frame,
        )

        # Press Q to exit.
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()