# Sign Language to Text Conversion System

## 📌 Project Overview


This project aims to bridge the communication gap between hearing-impaired individuals and others by converting hand sign gestures into text in real time. The system uses a webcam to capture hand movements, extracts hand landmarks using MediaPipe, and classifies gestures using a trained neural network.
=======
This project focuses on converting hand sign gestures into text in real time to help bridge the communication gap between hearing-impaired individuals and others. Using a webcam, the system captures hand movements, extracts hand landmarks, and classifies gestures using a trained neural network model.

The project involves end-to-end development including dataset creation, preprocessing, model training, real-time inference, and sentence formation.


---

## ✨ Features

- Real-time hand gesture recognition
- Letter-level sign language detection
- Sentence formation using temporal smoothing
- Custom dataset creation
- Balanced dataset for improved accuracy
- Extendable to words and full sentences
- Lightweight and runs on a standard webcam
=======
- Real-time hand gesture recognition using a webcam
- Letter-level sign language detection
- Sentence formation using temporal smoothing
- Custom dataset creation using hand landmarks
- Balanced dataset to improve classification accuracy
- Extendable architecture for adding more letters and words
- Lightweight and runs on standard hardware


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
=======
## ⚙️ System Workflow

1. Webcam captures live video frames  
2. MediaPipe detects the hand and extracts 21 landmarks  
3. Landmarks are converted into numerical feature vectors  
4. Features are passed to a trained neural network  
5. The model predicts the corresponding letter  
6. Temporal smoothing stabilizes predictions  
7. Letters are appended to form words and sentences  
>>>>>>> 390045e (Add detailed README documentation)

---

## 📊 Dataset Details

- Custom dataset created using MediaPipe hand landmarks

- Each gesture consists of 63 numerical features
- Dataset is manually balanced to avoid class bias
- Current supported letters:
  - **A, B, C, H, I**

> Note: The dataset and trained model are not included in this repository.
=======
- Each gesture sample contains 63 features (x, y, z for 21 landmarks)
- Dataset is manually balanced to prevent class bias
- Currently supported letters:
  - **A**
  - **B**
  - **C**
  - **H**
  - **I**

> Note: The dataset and trained model are intentionally excluded from this repository.


---

## ▶️ How to Run the Project

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
<<<<<<< HEAD
=======

##2 Collect Dataset
python src/collect_data.py

##3.Train the Model
python src/train_model.py

##4.Run live Prediction
python src/live_prediction.py


