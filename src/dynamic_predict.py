import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import time

SEQ_LEN = 30
model = tf.keras.models.load_model("../model/dynamic_model.h5")
labels = np.load("../model/dynamic_labels.npy", allow_pickle=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
cap = cv2.VideoCapture(0)

sequence = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        lm = result.multi_hand_landmarks[0]
        frame_data = []
        for p in lm.landmark:
            frame_data.extend([p.x, p.y, p.z])

        sequence.append(frame_data)

        if len(sequence) == SEQ_LEN:
            pred = model.predict(
                np.expand_dims(sequence, axis=0),
                verbose=0
            )
            word = labels[np.argmax(pred)]
            cv2.putText(frame, word, (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 3)
            sequence.pop(0)

    cv2.imshow("Dynamic Prediction", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

