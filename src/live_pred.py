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
ACCENT_GREEN = (0, 180, 0)
BAR_BG = (70, 70, 70)
BAR_FILL = (0, 200, 0)
SENTENCE_COLOR = (0, 120, 120)

# ==============================
# PREDICTION SETTINGS
# ==============================
CONF_THRESHOLD = 0.75
SMOOTH_FRAMES = 6
buffer = deque(maxlen=SMOOTH_FRAMES)

sentence = ""
last_char = ""
last_seen_time = time.time()
AUTO_SPACE_TIME = 2.0

# ==============================
# WORD SUGGESTIONS
# ==============================
WORD_LIST = [
    "HELLO", "HI", "HOW", "ARE", "YOU",
    "BYE", "THANK", "THANKYOU", "YES", "NO",
    "WHAT", "YOUR", "NAME"
]

def get_suggestions(text):
    last_word = text.strip().split(" ")[-1]
    if not last_word:
        return []
    return [w for w in WORD_LIST if w.startswith(last_word.upper())][:3]

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
    final_char = ""

    # ==============================
    # HEADER
    # ==============================
    cv2.rectangle(frame, (0, 0), (w, 55), HEADER_BG, -1)
    cv2.putText(frame, "SIGN LANGUAGE TO TEXT SYSTEM",
                (20, 38), FONT, 0.9, TEXT_WHITE, 2)

    # ==============================
    # GUIDE BOX (ROI)
    # ==============================
    BOX_X1, BOX_Y1 = 120, 80
    BOX_X2, BOX_Y2 = w - 120, h - 180

    cv2.rectangle(frame, (BOX_X1, BOX_Y1), (BOX_X2, BOX_Y2),
                  ACCENT_GREEN, 2)
    cv2.putText(frame, "Place hand inside the box",
                (BOX_X1 + 10, BOX_Y1 - 10),
                FONT, 0.6, ACCENT_GREEN, 2)

    # ==============================
    # HAND DETECTION (INSIDE BOX ONLY)
    # ==============================
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:

            inside_count = 0
            total_landmarks = len(hand_landmarks.landmark)

            for lm in hand_landmarks.landmark:
                px = int(lm.x * w)
                py = int(lm.y * h)

                if BOX_X1 <= px <= BOX_X2 and BOX_Y1 <= py <= BOX_Y2:
                    inside_count += 1

            # Require majority of landmarks inside box
            if inside_count / total_landmarks < 0.8:
                cv2.putText(frame, "Move hand inside box",
                            (20, 95), FONT, 0.8, (0, 0, 255), 2)
                continue

            # Accept ONLY one valid hand
            last_seen_time = time.time()
            mp_draw.draw_landmarks(frame, hand_landmarks,
                                   mp_hands.HAND_CONNECTIONS)

            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            landmarks = np.array(landmarks).reshape(1, -1)
            preds = model.predict(landmarks, verbose=0)

            confidence = np.max(preds)
            idx = np.argmax(preds)

            if idx < len(labels) and confidence >= CONF_THRESHOLD:
                predicted_char = labels[idx]
                buffer.append(predicted_char)

            break  # 🔒 ONLY ONE HAND

    # ==============================
    # TEMPORAL SMOOTHING
    # ==============================
    if len(buffer) == SMOOTH_FRAMES:
        if buffer.count(buffer[0]) == SMOOTH_FRAMES:
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
    # AUTO SPACE
    # ==============================
    if time.time() - last_seen_time > AUTO_SPACE_TIME:
        if sentence and not sentence.endswith(" "):
            sentence += " "
        last_seen_time = time.time()

    # ==============================
    # STATUS PANEL
    # ==============================
    cv2.rectangle(frame, (0, h - 160), (w, h - 55), PANEL_BG, -1)

    cv2.putText(frame, f"Detected Letter: {final_char}",
                (20, h - 120), FONT, 0.9, ACCENT_GREEN, 2)

    # ==============================
    # CONFIDENCE BAR
    # ==============================
    bar_x, bar_y = 20, h - 95
    bar_w, bar_h = 200, 20

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
    # WORD SUGGESTIONS
    # ==============================
    suggestions = get_suggestions(sentence)
    if suggestions:
        cv2.putText(frame, "Suggestions:",
                    (w - 320, h - 120),
                    FONT, 0.7, TEXT_WHITE, 2)
        for i, word in enumerate(suggestions):
            cv2.putText(frame, word,
                        (w - 320, h - 90 + i * 25),
                        FONT, 0.7, ACCENT_GREEN, 2)

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

