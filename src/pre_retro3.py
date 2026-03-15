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
HEADER_BG = (35,35,35)
PANEL_BG = (45,45,45)
TEXT_WHITE = (245,245,245)
ACCENT_GREEN = (0,200,0)
WARNING_RED = (0,0,255)
BAR_BG = (70,70,70)
BAR_FILL = (0,200,0)
SENTENCE_COLOR = (0,180,120)

# ==============================
# PREDICTION SETTINGS
# ==============================
CONF_THRESHOLD = 0.75
SMOOTH_FRAMES = 8

buffer = deque(maxlen=SMOOTH_FRAMES)
sentence = ""
last_char = ""

# ==============================
# CAMERA
# ==============================
cap = cv2.VideoCapture(0)

# ==============================
# TERMINAL BOOT SEQUENCE
# ==============================
boot_lines = [
">>> INITIALIZING NEURAL ENGINE",
">>> LOADING SIGN LANGUAGE MODEL",
">>> STARTING HAND TRACKING SYSTEM",
">>> CALIBRATING CAMERA INPUT",
">>> SYSTEM READY"
]

boot_frame = np.zeros((800,1280,3),dtype=np.uint8)

y = 120

for line in boot_lines:

    text=""

    for char in line:

        text += char
        boot_frame[:] = (10,0,30)

        yy = 120

        for prev in boot_lines[:boot_lines.index(line)]:
            cv2.putText(boot_frame,prev,(120,yy),FONT,0.9,(0,255,0),2)
            yy += 60

        cv2.putText(boot_frame,text,(120,y),FONT,0.9,(0,255,0),2)

        for s in range(0,800,3):
            cv2.line(boot_frame,(0,s),(1280,s),(15,0,40),1)

        cv2.imshow("Sign Language Recognition",boot_frame)
        cv2.waitKey(30)

    y += 60
    time.sleep(0.2)

time.sleep(1)

# ==============================
# RETRO INTRO
# ==============================
ret, cam = cap.read()
h,w,_ = cam.shape

stars = np.random.randint(0,[w,h],(150,2))
grid_offset = 0
horizon = h//2

while True:

    frame = np.zeros((h,w,3),dtype=np.uint8)
    frame[:] = (10,0,30)

    for star in stars:
        cv2.circle(frame,(star[0],star[1]),1,(255,255,255),-1)
        star[1]+=2
        if star[1] > h:
            star[0]=np.random.randint(0,w)
            star[1]=0

    cv2.rectangle(frame,(0,horizon),(w,horizon+4),(255,0,255),-1)

    grid_offset+=0.02
    num_lines=25

    for i in range(num_lines):

        depth=(i+grid_offset)%num_lines
        perspective=depth/num_lines

        y_line=int(horizon+perspective*(h-horizon))
        width=int(w*perspective*0.8)

        x1=w//2-width
        x2=w//2+width

        cv2.line(frame,(x1,y_line),(x2,y_line),(255,0,255),2)

    for i in range(-7,7):
        x_top=w//2+i*40
        x_bottom=w//2+i*120
        cv2.line(frame,(x_top,horizon),(x_bottom,h),(255,0,255),1)

    title="SIGN LANGUAGE AI"
    (tw,th),_ = cv2.getTextSize(title,FONT,2.2,3)
    title_x=(w-tw)//2
    title_y=horizon-80

    for glow in range(10,0,-1):
        cv2.putText(frame,title,(title_x,title_y),FONT,2.2,(255,0,255),glow*2)

    cv2.putText(frame,title,(title_x,title_y),FONT,2.2,(255,255,255),3)

    subtitle="REAL TIME SIGN LANGUAGE RECOGNITION"
    (sw,sh),_ = cv2.getTextSize(subtitle,FONT,0.9,2)

    cv2.putText(frame,subtitle,((w-sw)//2,horizon-20),FONT,0.9,(255,255,0),2)

    for s in range(0,h,2):
        cv2.line(frame,(0,s),(w,s),(15,0,40),1)

    start_text="PRESS ENTER TO START"
    (stw,sth),_ = cv2.getTextSize(start_text,FONT,1.0,2)

    if int(time.time()*2)%2==0:
        cv2.putText(frame,start_text,((w-stw)//2,horizon+140),FONT,1.0,(0,255,120),2)

    quit_text="PRESS Q TO QUIT"
    (qw,qh),_ = cv2.getTextSize(quit_text,FONT,0.7,1)

    cv2.putText(frame,quit_text,((w-qw)//2,horizon+190),FONT,0.7,(200,200,200),1)

    cv2.imshow("Sign Language Recognition",frame)

    key=cv2.waitKey(1)&0xFF

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

    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    result=hands.process(rgb)

    predicted_char=""
    confidence=0.0
    warning_text=None

    cv2.rectangle(frame,(0,0),(w,55),HEADER_BG,-1)

    cv2.putText(frame,"SIGN LANGUAGE TO TEXT SYSTEM",(20,38),FONT,0.9,TEXT_WHITE,2)

    cv2.putText(frame,"MODE: LIVE DETECTION",(w-230,35),FONT,0.6,(180,180,180),1)

    roi_x1,roi_y1=120,80
    roi_x2,roi_y2=w-120,h-180

    cv2.rectangle(frame,(roi_x1,roi_y1),(roi_x2,roi_y2),ACCENT_GREEN,2)

    cv2.putText(frame,"Hold hand still inside the box",(roi_x1,roi_y1-10),FONT,0.6,ACCENT_GREEN,2)

    if result.multi_hand_landmarks:

        hand = result.multi_hand_landmarks[0]

        all_inside = True

        for lm in hand.landmark:

            px = int(lm.x * w)
            py = int(lm.y * h)

            if px < roi_x1 or px > roi_x2 or py < roi_y1 or py > roi_y2:
                all_inside = False
                break

        if all_inside:

            mp_draw.draw_landmarks(frame,hand,mp_hands.HAND_CONNECTIONS)

            landmarks=np.array([[lm.x,lm.y,lm.z] for lm in hand.landmark]).flatten()

            preds=model.predict(landmarks.reshape(1,-1),verbose=0)

            confidence=np.max(preds)
            idx=np.argmax(preds)

            predicted_char=labels[idx]

            buffer.append(predicted_char)

        else:
            warning_text="Place your FULL hand inside the box"
            buffer.clear()

    else:
        warning_text="No hand detected"
        buffer.clear()

    final_char=""

    if len(buffer)==SMOOTH_FRAMES and buffer.count(buffer[0])==SMOOTH_FRAMES:
        final_char=buffer[0]

    if final_char and final_char!=last_char:
        sentence+=final_char
        last_char=final_char

    if not final_char:
        last_char=""

    if warning_text:
        (tw,th),_ = cv2.getTextSize(warning_text,FONT,0.8,2)
        cv2.rectangle(frame,(0,55),(w,95),(0,0,0),-1)
        cv2.putText(frame,warning_text,((w-tw)//2,85),FONT,0.8,WARNING_RED,2)

    cv2.rectangle(frame,(0,h-160),(w,h-55),PANEL_BG,-1)

    cv2.putText(frame,f"Detected: {final_char}",(20,h-120),FONT,0.9,ACCENT_GREEN,2)

    bar_x,bar_y,bar_w,bar_h=20,h-95,200,20

    cv2.rectangle(frame,(bar_x,bar_y),(bar_x+bar_w,bar_y+bar_h),BAR_BG,-1)

    cv2.rectangle(frame,(bar_x,bar_y),(bar_x+int(bar_w*confidence),bar_y+bar_h),BAR_FILL,-1)

    cv2.putText(frame,f"{int(confidence*100)}%",(bar_x+bar_w+10,bar_y+17),FONT,0.6,TEXT_WHITE,1)

    cv2.rectangle(frame,(10,h-50),(w-10,h-10),(30,30,30),-1)

    cursor="|" if int(time.time()*2)%2==0 else ""

    cv2.putText(frame,f"Sentence: {sentence}{cursor}",(20,h-20),FONT,0.8,SENTENCE_COLOR,2)

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
