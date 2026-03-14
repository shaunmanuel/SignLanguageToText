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
SENTENCE_COLOR = (0, 180, 120)

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

# ==============================
# TERMINAL BOOT SEQUENCE
# ==============================
boot_lines = [
    "Initializing Neural Engine...",
    "Loading Gesture Recognition Model...",
    "Starting Hand Tracking System...",
    "Calibrating Camera Input...",
    "Allocating GPU Memory...",
    "System Ready."
]

boot_frame = np.zeros((480,800,3),dtype=np.uint8)

y = 60
for line in boot_lines:

    text = ""
    for char in line:
        text += char

        boot_frame[:] = (0,0,0)

        yy = 60
        for prev in boot_lines[:boot_lines.index(line)]:
            cv2.putText(boot_frame,prev,(40,yy),FONT,0.7,(0,255,0),2)
            yy += 40

        cv2.putText(boot_frame,text,(40,y),FONT,0.7,(0,255,0),2)

        cv2.imshow("Sign Language Recognition",boot_frame)
        cv2.waitKey(40)

    y += 40
    time.sleep(0.3)

time.sleep(1)

# ==============================
# RETRO SYNTHWAVE INTRO SCREEN
# ==============================
grid_offset = 0

while True:

    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    frame[:] = (25,0,40)

    grid_offset += 2

    for i in range(0,w,40):
        cv2.line(frame,(i,h//2),(w//2,h),(255,0,255),1)

    for j in range(0,h,40):
        y = (j+grid_offset)%h
        cv2.line(frame,(0,y),(w,y),(180,0,255),1)

    title = "SIGN LANGUAGE AI"

    for i in range(8,0,-1):
        cv2.putText(frame,title,(w//2-280,h//2-40),
                    FONT,1.8,(255,0,255),i*2)

    cv2.putText(frame,title,(w//2-280,h//2-40),
                FONT,1.8,(255,255,255),2)

    cv2.putText(frame,
        "REAL TIME SIGN LANGUAGE RECOGNITION",
        (w//2-260,h//2+20),
        FONT,0.7,(255,255,0),2)

    for yy in range(0,h,3):
        cv2.line(frame,(0,yy),(w,yy),(10,0,20),1)

    if int(time.time()*2)%2==0:
        cv2.putText(frame,
            "PRESS ENTER TO START",
            (w//2-200,h//2+100),
            FONT,0.9,(0,255,120),2)

    cv2.putText(frame,
        "PRESS Q TO QUIT",
        (w//2-120,h//2+140),
        FONT,0.6,(200,200,200),1)

    cv2.imshow("Sign Language Recognition",frame)

    key = cv2.waitKey(1)&0xFF

    if key==13:
        break

    if key==ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        exit()

# ==============================
# MAIN DETECTION LOOP
# ==============================
while True:

    ret, frame = cap.read()
    if not ret:
        break

    h,w,_ = frame.shape

    rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    predicted_char = ""
    confidence = 0.0

    # HEADER
    cv2.rectangle(frame,(0,0),(w,55),HEADER_BG,-1)

    cv2.putText(frame,"SIGN LANGUAGE TO TEXT SYSTEM",
                (20,38),FONT,0.9,TEXT_WHITE,2)

    cv2.putText(frame,"MODE: LIVE DETECTION",
                (w-230,35),FONT,0.6,(180,180,180),1)

    # ROI BOX
    roi_x1,roi_y1 = 120,80
    roi_x2,roi_y2 = w-120,h-180

    cv2.rectangle(frame,(roi_x1,roi_y1),(roi_x2,roi_y2),
                  ACCENT_GREEN,2)

    cv2.putText(frame,"Hold hand still inside the box",
                (roi_x1,roi_y1-10),
                FONT,0.6,ACCENT_GREEN,2)

    # HAND DETECTION
    if result.multi_hand_landmarks:

        hand = result.multi_hand_landmarks[0]

        cx = int(hand.landmark[0].x*w)
        cy = int(hand.landmark[0].y*h)

        if roi_x1<cx<roi_x2 and roi_y1<cy<roi_y2:

            mp_draw.draw_landmarks(frame,hand,
                                   mp_hands.HAND_CONNECTIONS)

            last_seen_time = time.time()

            landmarks = np.array([[lm.x,lm.y,lm.z]
                        for lm in hand.landmark]).flatten()

            is_stable = True

            if prev_landmarks is not None:

                motion = np.mean(np.abs(landmarks-prev_landmarks))

                if motion>MOTION_THRESHOLD:
                    is_stable=False
                    stable_start_time=None
            else:
                is_stable=False

            prev_landmarks = landmarks

            preds = model.predict(landmarks.reshape(1,-1),
                                  verbose=0)

            confidence = np.max(preds)
            idx = np.argmax(preds)

            now = time.time()

            if confidence>=CONF_THRESHOLD and is_stable:

                if stable_start_time is None:
                    stable_start_time = now

                if now-stable_start_time>=STABLE_TIME_REQUIRED:
                    predicted_char = labels[idx]
                    buffer.append(predicted_char)

            else:
                stable_start_time=None
                buffer.clear()

        else:

            warn_text="⚠ Place your hand INSIDE the box"

            cv2.rectangle(frame,(0,55),(w,95),(0,0,0),-1)

            (tw,th),_ = cv2.getTextSize(warn_text,
                                       FONT,0.8,2)

            cv2.putText(frame,warn_text,
                        ((w-tw)//2,85),
                        FONT,0.8,WARNING_RED,2)

            prev_landmarks=None
            buffer.clear()
            stable_start_time=None

    else:
        prev_landmarks=None
        buffer.clear()
        stable_start_time=None

    # HOLD PROGRESS BAR
    if stable_start_time is not None:
        progress = min((time.time()-stable_start_time)
                       /STABLE_TIME_REQUIRED,1)
    else:
        progress=0

    bar_center=w//2
    bar_y=roi_y2+15
    bar_width=200
    bar_height=18

    cv2.rectangle(frame,(bar_center-bar_width//2,bar_y),
        (bar_center+bar_width//2,bar_y+bar_height),
        BAR_BG,-1)

    cv2.rectangle(frame,(bar_center-bar_width//2,bar_y),
        (bar_center-bar_width//2+int(bar_width*progress),
        bar_y+bar_height),
        ACCENT_GREEN,-1)

    cv2.putText(frame,"Hold Gesture",
        (bar_center-60,bar_y-5),
        FONT,0.5,TEXT_WHITE,1)

    # TEMPORAL SMOOTHING
    final_char=""

    if len(buffer)==SMOOTH_FRAMES and buffer.count(buffer[0])==SMOOTH_FRAMES:
        final_char=buffer[0]

    # SENTENCE
    if final_char and final_char!=last_char:
        sentence+=final_char
        last_char=final_char

    if not final_char:
        last_char=""

    if time.time()-last_seen_time>AUTO_SPACE_TIME:

        if sentence and not sentence.endswith(" "):
            sentence+=" "

        last_seen_time=time.time()

    # STATUS PANEL
    cv2.rectangle(frame,(0,h-160),(w,h-55),PANEL_BG,-1)

    cv2.putText(frame,f"Detected: {final_char}",
        (20,h-120),FONT,0.9,ACCENT_GREEN,2)

    # CONFIDENCE BAR
    bar_x,bar_y,bar_w,bar_h=20,h-95,200,20

    cv2.rectangle(frame,(bar_x,bar_y),
        (bar_x+bar_w,bar_y+bar_h),
        BAR_BG,-1)

    cv2.rectangle(frame,(bar_x,bar_y),
        (bar_x+int(bar_w*confidence),bar_y+bar_h),
        BAR_FILL,-1)

    cv2.putText(frame,f"{int(confidence*100)}%",
        (bar_x+bar_w+10,bar_y+17),
        FONT,0.6,TEXT_WHITE,1)

    # SENTENCE BOX
    cv2.rectangle(frame,(10,h-50),(w-10,h-10),(30,30,30),-1)

    cursor="|" if int(time.time()*2)%2==0 else ""

    cv2.putText(frame,f"Sentence: {sentence}{cursor}",
        (20,h-20),FONT,0.8,SENTENCE_COLOR,2)

    # FOOTER
    cv2.putText(frame,
        "Q: Quit | C: Clear | D: Delete | SPACE: Space",
        (w-520,38),FONT,0.55,TEXT_WHITE,1)

    cv2.imshow("Sign Language Recognition",frame)

    key=cv2.waitKey(1)&0xFF

    if key==ord('q'):
        break
    elif key==ord('c'):
        sentence=""
    elif key==ord('d'):
        sentence=sentence[:-1]
    elif key==ord(' '):
        sentence+=" "

cap.release()
cv2.destroyAllWindows()
