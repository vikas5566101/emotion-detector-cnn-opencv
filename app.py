import gradio as gr
import cv2
import numpy as np
from pathlib import Path
from tensorflow.keras.models import load_model

# Project paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.keras"

# Emotion classes in the model's output order
EMOTION_LABELS = [
    "Angry", "Disgust", "Fear", "Happy",
    "Neutral", "Sad", "Surprise"
]

# Load the trained model and face detector
model = load_model(str(MODEL_PATH))

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def detect_emotion(image):
    if image is None:
        return None, "Please enable your webcam."

    # Gradio supplies an RGB image
    frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30, 30)
    )

    results = []

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))

        # Match the preprocessing used by the working local version
        roi = face.astype(np.float32)
        roi = np.expand_dims(roi, axis=-1)
        roi = np.expand_dims(roi, axis=0)

        predictions = model.predict(roi, verbose=0)[0]
        emotion_index = int(np.argmax(predictions))
        emotion = EMOTION_LABELS[emotion_index]
        confidence = float(predictions[emotion_index]) * 100

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        cv2.putText(
            frame,
            f"{emotion} ({confidence:.1f}%)",
            (x, max(y-10, 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        results.append(f"{emotion}: {confidence:.1f}%")

    output = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if not results:
        return output, "No face detected."

    return output, "\n".join(results)

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
        "Detect facial expressions using a trained CNN, "
        "TensorFlow/Keras, and OpenCV."
    )
)


import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port
    )

