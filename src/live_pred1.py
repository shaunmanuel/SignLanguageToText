import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import os
import time
from collections import deque

# ==============================
# PATH SETUP
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../model/gesture_model.h5")
LABELS_PATH = os.path.join(BASE_DIR, "../model/labels.npy")

model = tf.keras.models.load_model(MODEL_PATH)
labels = np.load(LABELS_PATH, allow_pickle=True)

# ==============================
# MEDIAPIPE
# ==============================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# ==============================
# UI CONFIG
# ==============================
FONT = cv2.FONT_HERSHEY_SIMPLEX
HEADER_BG = (35, 35, 35)
PANEL_BG = (45, 45, 45)
TEXT_WHITE = (245, 245, 245)
ACCENT_GREEN = (0, 200, 0)
WARNING_RED = (0, 0, 255)
BAR_BG = (70, 70, 70)
BAR_FILL = (0, 200, 0)
SENTENCE_COLOR = (0, 180, 120)  # dark, readable green

# ==============================
# PREDICTION SETTINGS
# ==============================
CONF_THRESHOLD = 0.75
SMOOTH_FRAMES = 8
STABLE_TIME_REQUIRED = 0.4
MOTION_THRESHOLD = 0.015

buffer = deque(maxlen=SMOOTH_FRAMES)
sentence = ""
last_char = ""
last_seen_time = time.time()
AUTO_SPACE_TIME = 1.5

prev_landmarks = None
stable_start_time = None

# ==============================
# CAMERA
# ==============================
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    predicted_char = ""
    confidence = 0.0

    # ==============================
    # HEADER
    # ==============================
    cv2.rectangle(frame, (0, 0), (w, 55), HEADER_BG, -1)
    cv2.putText(frame, "SIGN LANGUAGE TO TEXT SYSTEM",
                (20, 38), FONT, 0.9, TEXT_WHITE, 2)

    # ==============================
    # GUIDE BOX (ROI)
    # ==============================
    roi_x1, roi_y1 = 120, 80
    roi_x2, roi_y2 = w - 120, h - 180
    cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), ACCENT_GREEN, 2)
    cv2.putText(frame, "Hold hand still inside the box",
                (roi_x1, roi_y1 - 10), FONT, 0.6, ACCENT_GREEN, 2)

    # ==============================
    # HAND DETECTION
    # ==============================
    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]

        # Hand center (wrist) in pixel coords
        cx = int(hand.landmark[0].x * w)
        cy = int(hand.landmark[0].y * h)

        # Check ROI
        if roi_x1 < cx < roi_x2 and roi_y1 < cy < roi_y2:
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
            last_seen_time = time.time()

            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark]).flatten()

            # ------------------------------
            # MOTION CHECK (block movement)
            # ------------------------------
            is_stable = True
            if prev_landmarks is not None:
                motion = np.mean(np.abs(landmarks - prev_landmarks))
                if motion > MOTION_THRESHOLD:
                    is_stable = False
                    stable_start_time = None
            else:
                is_stable = False

            prev_landmarks = landmarks

            # ------------------------------
            # PREDICTION
            # ------------------------------
            preds = model.predict(landmarks.reshape(1, -1), verbose=0)
            confidence = np.max(preds)
            idx = np.argmax(preds)
            now = time.time()

            if confidence >= CONF_THRESHOLD and is_stable:
                if stable_start_time is None:
                    stable_start_time = now
                if now - stable_start_time >= STABLE_TIME_REQUIRED:
                    predicted_char = labels[idx]
                    buffer.append(predicted_char)
            else:
                stable_start_time = None
                buffer.clear()

        else:
            # ==============================
            # TOP-CENTER WARNING (OUTSIDE ROI)
            # ==============================
            warn_text = "⚠ Place your hand INSIDE the box"
            cv2.rectangle(frame, (0, 55), (w, 95), (0, 0, 0), -1)
            (tw, th), _ = cv2.getTextSize(warn_text, FONT, 0.8, 2)
            cv2.putText(frame, warn_text,
                        ((w - tw)//2, 85),
                        FONT, 0.8, WARNING_RED, 2)

            prev_landmarks = None
            buffer.clear()
            stable_start_time = None

    else:
        prev_landmarks = None
        buffer.clear()
        stable_start_time = None

    # ==============================
    # TEMPORAL SMOOTHING
    # ==============================
    final_char = ""
    if len(buffer) == SMOOTH_FRAMES and buffer.count(buffer[0]) == SMOOTH_FRAMES:
        final_char = buffer[0]

    # ==============================
    # SENTENCE FORMATION
    # ==============================
    if final_char and final_char != last_char:
        sentence += final_char
        last_char = final_char
    if not final_char:
        last_char = ""

    # ==============================
    # AUTO SPACE AFTER PAUSE
    # ==============================
    if time.time() - last_seen_time > AUTO_SPACE_TIME:
        if sentence and not sentence.endswith(" "):
            sentence += " "
        last_seen_time = time.time()

    # ==============================
    # STATUS PANEL
    # ==============================
    cv2.rectangle(frame, (0, h - 160), (w, h - 55), PANEL_BG, -1)
    cv2.putText(frame, f"Detected: {final_char}",
                (20, h - 120), FONT, 0.9, ACCENT_GREEN, 2)

    # ==============================
    # CONFIDENCE BAR
    # ==============================
    bar_x, bar_y, bar_w, bar_h = 20, h - 95, 200, 20
    cv2.rectangle(frame, (bar_x, bar_y),
                  (bar_x + bar_w, bar_y + bar_h), BAR_BG, -1)
    cv2.rectangle(frame, (bar_x, bar_y),
                  (bar_x + int(bar_w * confidence), bar_y + bar_h),
                  BAR_FILL, -1)
    cv2.putText(frame, f"{int(confidence * 100)}%",
                (bar_x + bar_w + 10, bar_y + 17),
                FONT, 0.6, TEXT_WHITE, 1)

    # ==============================
    # SENTENCE DISPLAY
    # ==============================
    cv2.putText(frame, f"Sentence: {sentence}",
                (20, h - 30), FONT, 0.9, SENTENCE_COLOR, 2)

    # ==============================
    # FOOTER
    # ==============================
    cv2.putText(frame,
                "Q: Quit | C: Clear | D: Delete | SPACE: Space",
                (w - 520, 38), FONT, 0.55, TEXT_WHITE, 1)

    cv2.imshow("Sign Language Recognition", frame)

    # ==============================
    # KEYS
    # ==============================
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        sentence = ""
    elif key == ord('d'):
        sentence = sentence[:-1]
    elif key == ord(' '):
        sentence += " "

cap.release()
cv2.destroyAllWindows()
