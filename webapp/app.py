from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2
import base64
import mediapipe as mp
import tensorflow as tf
import os

app = Flask(__name__)

# ✅ Safe model path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "..", "src", "model", "gesture_model.h5")

# Load trained gesture model
model = tf.keras.models.load_model(model_path)

labels = ["A","B","C","D","E","F","G","H","I","J",
          "K","L","M","N","O","P","Q","R","S","T",
          "U","V","W","X","Y","Z"]

# ✅ Proper MediaPipe init
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["image"]

    # Decode base64 image
    img_bytes = base64.b64decode(data.split(',')[1])
    np_arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Convert to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # MediaPipe detection
    result = hands.process(frame_rgb)

    gesture = "No hand detected"

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]

        landmark_data = []
        for lm in hand_landmarks.landmark:
            landmark_data.extend([lm.x, lm.y, lm.z])

        landmark_data = np.array(landmark_data).reshape(1, -1)

        prediction = model.predict(landmark_data, verbose=0)
        gesture = labels[np.argmax(prediction)]

    return jsonify({"prediction": gesture})

if __name__ == "__main__":
    # ✅ Disable Flask auto-reloader (prevents double tabs & camera issues)
    app.run(debug=True, use_reloader=False)