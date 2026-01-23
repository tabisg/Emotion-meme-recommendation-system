# 🎭 Real-Time Emotion Detection System

A real-time AI-based computer vision project that detects human facial emotions from live webcam input using deep learning.  
The system analyzes facial expressions and displays the detected emotion along with basic emotion analytics.

---

## 📖 Project Overview

Human emotions play a key role in communication, but traditional computer systems lack emotional awareness.  
This project bridges that gap by using **Artificial Intelligence and Computer Vision** to detect facial emotions in real time.

The system captures live video from a webcam, analyzes facial expressions using a deep learning model, and identifies the dominant emotion.

---

## 🚀 Key Features

- Real-time facial emotion detection  
- Webcam-based live video processing  
- Deep learning-based emotion classification  
- Emotion confidence score  
- Emotion logging with timestamps  
- Basic emotion analytics visualization  

---

## 🧠 Emotions Detected

The system detects the following emotions:
- Happy  
- Sad  
- Angry  
- Fear  
- Surprise  
- Disgust  
- Neutral  

---

## 🛠️ Technologies Used

| Category | Technology |
|--------|------------|
| Programming Language | Python |
| Computer Vision | OpenCV |
| Deep Learning | DeepFace (CNN-based models) |
| Data Handling | NumPy, CSV |
| Visualization | Matplotlib |

---

## ⚙️ How the System Works

1. Webcam captures live video frames  
2. Frames are processed using OpenCV  
3. Facial features are analyzed using DeepFace  
4. The dominant emotion is detected with a confidence score  
5. Emotion data is logged with timestamps  
6. Emotion frequency analytics are generated after execution  

---

## ▶️ How to Run the Project

### Prerequisites
- Python 3.9+
- Webcam

### Install Dependencies
```bash
pip install opencv-python numpy deepface matplotlib
