# CNN-Based Facial Emotion Detection

A deep learning application that detects faces in images and predicts facial expressions using a Convolutional Neural Network (CNN), TensorFlow/Keras, and OpenCV. The application provides a web interface built with Gradio.

## Features

- **Face Detection:** Detects faces using OpenCV's Haar Cascade classifier.
- **Emotion Classification:** Predicts one of seven facial expression categories using a trained CNN.
- **Confidence Scores:** Displays the model's confidence score for each detected face.
- **Face Visualization:** Draws bounding boxes and predicted emotion labels around detected faces.
- **Web Interface:** Provides an interactive interface using Gradio.
- **Local Execution:** Supports running the application on your local machine.

## Supported Emotions

The model classifies facial expressions into seven categories:

- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Application development |
| TensorFlow / Keras | CNN model loading and inference |
| OpenCV | Face detection and image processing |
| NumPy | Numerical operations and image preparation |
| Gradio | Web-based user interface |

## Project Structure

```text
emotion-detector-cnn-opencv/
├── app.py
├── main.py
├── model.keras
├── cnn.ipynb
├── requirements.txt
└── README.md
```

- `app.py` — Gradio web application for face detection and emotion prediction.
- `main.py` — Local application entry point.
- `model.keras` — Trained CNN model.
- `cnn.ipynb` — Notebook containing the model development and training workflow.
- `requirements.txt` — Python dependencies.
- `README.md` — Project documentation.

## How It Works

The application follows this workflow:

1. **Image Input:** The user provides an image through the webcam interface.
2. **Image Preprocessing:** The input image is converted from RGB to grayscale.
3. **Face Detection:** OpenCV's Haar Cascade classifier identifies facial regions.
4. **Image Resizing:** Each detected face is resized to `48 × 48` pixels.
5. **CNN Inference:** The preprocessed facial image is passed to the trained Keras model.
6. **Emotion Prediction:** The model produces scores for the seven emotion categories, and the category with the highest score is selected.
7. **Result Visualization:** The application displays bounding boxes, emotion labels, and confidence scores.

## Installation and Setup

### Prerequisites

- Python 3.10 or 3.11 recommended
- Git
- A webcam, or a compatible image input

### 1. Clone the repository

```bash
git clone https://github.com/vikas5566101/emotion-detector-cnn-opencv.git
cd emotion-detector-cnn-opencv
```

### 2. Create a virtual environment

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the application

```bash
python app.py
```

Open the local URL printed in the terminal by Gradio, usually:

```text
http://127.0.0.1:7860
```

Allow webcam access if your browser requests permission.

## Model Information

- **Architecture:** Convolutional Neural Network (CNN)
- **Framework:** TensorFlow/Keras
- **Input shape:** `48 × 48 × 1`
- **Input format:** Grayscale facial image
- **Output classes:** 7 facial expression categories

The trained model is stored in `model.keras` and loaded when the application starts.

## Dataset

The model was trained using the [Face Expression Recognition Dataset](https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset) available on Kaggle.

The dataset contains facial images organized into expression categories. Refer to the training notebook for the dataset loading, preprocessing, and model training workflow.

## Limitations

- Emotion predictions are estimates of visible facial expressions and do not establish a person's actual emotional state.
- Prediction quality can be affected by lighting, face orientation, image quality, occlusion, and differences between training and real-world images.
- The model supports the seven categories listed above and does not identify every possible facial expression.
- Cloud hosting may require additional memory optimization because TensorFlow inference can be resource-intensive on low-memory instances.

## Future Improvements

- Evaluate the model on a separate test set and report verified classification metrics.
- Improve inference efficiency for resource-constrained environments.
- Explore additional face detection and model optimization techniques.
- Improve robustness across lighting conditions and face orientations.

## Author

**Vikas Kumar Prajapati**

- GitHub: [@vikas5566101](https://github.com/vikas5566101)

---

*This project was developed for learning and demonstrating deep learning, computer vision, and web application integration.*