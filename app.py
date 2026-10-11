
import os

# Limit CPU threads to reduce resource usage
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import gc
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
from tensorflow.keras.models import load_model

# Project paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.keras"

EMOTION_LABELS = [
    "Angry", "Disgust", "Fear", "Happy",
    "Neutral", "Sad", "Surprise"
]

# Load the model only once, without training-related configuration
model = load_model(str(MODEL_PATH), compile=False)

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def detect_emotion(image):
    if image is None:
        return None, "Please capture an image using your webcam."

    # Work with one image at a time
    frame = np.asarray(image, dtype=np.uint8).copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30, 30)
    )

    if len(faces) == 0:
        return frame, "No face detected."

    # Prepare all detected faces as one batch
    face_batch = []

    for (x, y, w, h) in faces:
        face = gray[y:y + h, x:x + w]
        face = cv2.resize(face, (48, 48))
        face = face.astype(np.float32)
        face_batch.append(face)

    # Shape: (number_of_faces, 48, 48, 1)
    face_batch = np.asarray(face_batch, dtype=np.float32)[..., np.newaxis]

    # Direct inference avoids model.predict()'s extra prediction machinery
    predictions = model(face_batch, training=False).numpy()

    results = []

    for (x, y, w, h), scores in zip(faces, predictions):
        emotion_index = int(np.argmax(scores))
        emotion = EMOTION_LABELS[emotion_index]
        confidence = float(scores[emotion_index]) * 100

        cv2.rectangle(
            frame, (x, y), (x + w, y + h), (0, 255, 255), 2
        )

        cv2.putText(
            frame,
            f"{emotion} ({confidence:.1f}%)",
            (x, max(y - 10, 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        results.append(f"{emotion}: {confidence:.1f}%")

    del face_batch, predictions
    gc.collect()

    return frame, "\n".join(results)


demo = gr.Interface(
    fn=detect_emotion,
    inputs=gr.Image(
        sources=["webcam"],
        type="numpy",
        label="Webcam"
    ),
    outputs=[
        gr.Image(label="Face Detection"),
        gr.Textbox(label="Emotion Prediction")
    ],
    title="CNN-Based Facial Emotion Detection",
    description=(
        "Capture an image using your webcam and click Submit "
        "to predict facial emotions using CNN, TensorFlow, and OpenCV."
    ),
    flagging_mode="never"
)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))

    demo.queue(default_concurrency_limit=1)

    demo.launch(
        server_name="0.0.0.0",
        server_port=port
    )
