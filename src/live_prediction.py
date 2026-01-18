import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from collections import deque

# ==============================
# LOAD TRAINED MODEL
# ==============================
model = tf.keras.models.load_model("../model/gesture_model.h5")

# Label mapping (must match training order)
labels = ["A", "B", "C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]

# ==============================
# MEDIAPIPE HANDS SETUP
# ==============================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# ==============================
# SMOOTHING & SENTENCE VARIABLES
# ==============================
prediction_buffer = deque(maxlen=10)   # temporal smoothing
sentence = ""
last_confirmed = ""

CONFIDENCE_THRESHOLD = 0.80

# ==============================
# COLORS (BGR FORMAT)
# ==============================
GESTURE_COLOR = (0, 255, 0)        # Green
SENTENCE_COLOR = (80, 140, 0)      # Medium-dark teal (recommended)

# ==============================
# WEBCAM START
# ==============================
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    current_gesture = ""
    confidence = 0.0

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

            # Extract 63 landmark features
            data = []
            for lm in hand_landmarks.landmark:
                data.extend([lm.x, lm.y, lm.z])

            data = np.array(data).reshape(1, -1)

            # Predict gesture
            prediction = model.predict(data, verbose=0)
            confidence = np.max(prediction)
            predicted_index = np.argmax(prediction)

            if confidence >= CONFIDENCE_THRESHOLD:
                current_gesture = labels[predicted_index]

    # ==============================
    # TEMPORAL SMOOTHING
    # ==============================
    if current_gesture != "":
        prediction_buffer.append(current_gesture)

    final_gesture = ""
    if len(prediction_buffer) == prediction_buffer.maxlen:
        if prediction_buffer.count(prediction_buffer[0]) == prediction_buffer.maxlen:
            final_gesture = prediction_buffer[0]

    # ==============================
    # CONFIRM LETTER ONCE
    # ==============================
    if final_gesture != "" and final_gesture != last_confirmed:
        sentence += final_gesture
        last_confirmed = final_gesture

    if final_gesture == "":
        last_confirmed = ""

    # ==============================
    # DISPLAY OUTPUT
    # ==============================
    cv2.putText(
        frame,
        f"Gesture: {final_gesture} ({confidence:.2f})",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        GESTURE_COLOR,
        2
    )

    cv2.putText(
        frame,
        f"Sentence: {sentence}",
        (30, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        SENTENCE_COLOR,
        2
    )

    cv2.imshow("Sign Language to Text (Press Q to Exit)", frame)

    # ==============================
    # KEYBOARD CONTROLS
    # ==============================
    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '):        # Space
        sentence += " "
    elif key == ord('d'):      # Delete last character
        sentence = sentence[:-1]
    elif key == ord('c'):      # Clear sentence
        sentence = ""
    elif key == ord('q'):      # Quit
        break

cap.release()
cv2.destroyAllWindows()

