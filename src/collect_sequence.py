import cv2
import mediapipe as mp
import numpy as np
import os

LABEL = input("Enter word label (e.g., HELLO): ").strip().upper()
SEQ_LEN = 30

base_dir = os.path.join("..", "data_dynamic", LABEL)
os.makedirs(base_dir, exist_ok=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
cap = cv2.VideoCapture(0)

sequence = []
count = 0

print("Perform the gesture repeatedly. Press Q to stop.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        landmarks = result.multi_hand_landmarks[0]
        frame_data = []
        for lm in landmarks.landmark:
            frame_data.extend([lm.x, lm.y, lm.z])

        sequence.append(frame_data)

        if len(sequence) == SEQ_LEN:
            np.save(
                os.path.join(base_dir, f"seq_{count}.npy"),
                np.array(sequence)
            )
            print(f"Saved sequence {count}")
            sequence = []
            count += 1

    cv2.imshow("Collect Dynamic Gesture", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

