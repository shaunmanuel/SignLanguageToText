# sign-recognition

# Sign Language to Text Conversion System

A real-time Sign Language to Text conversion system that recognizes hand gestures and converts them into readable text using Computer Vision and Machine Learning techniques.

---

## 📌 Project Overview

This project aims to bridge the communication gap between hearing-impaired individuals and others by converting hand sign gestures into text in real time. The system uses a webcam to capture hand movements, extracts hand landmarks using MediaPipe, and classifies gestures using a trained neural network.

---

## ✨ Features

- Real-time hand gesture recognition
- Letter-level sign language detection
- Sentence formation using temporal smoothing
- Custom dataset creation
- Balanced dataset for improved accuracy
- Extendable to words and full sentences
- Lightweight and runs on a standard webcam

---

## 🛠️ Tech Stack

- **Programming Language:** Python  
- **Computer Vision:** OpenCV  
- **Hand Landmark Detection:** MediaPipe  
- **Machine Learning:** TensorFlow / Keras  
- **Data Processing:** NumPy, Pandas  

---

## ⚙️ How the System Works

1. Webcam captures live video input  
2. MediaPipe extracts 21 hand landmarks (63 features)  
3. Landmarks are fed into a trained neural network  
4. The model predicts the corresponding letter  
5. Temporal smoothing stabilizes predictions  
6. Letters are combined to form words and sentences  

---

## 📊 Dataset Details

- Custom dataset created using MediaPipe hand landmarks
- Each gesture consists of 63 numerical features
- Dataset is manually balanced to avoid class bias
- Current supported letters:
  - **A, B, C, H, I**

> Note: The dataset and trained model are not included in this repository.

---

## ▶️ How to Run the Project

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
(Create README.md)
