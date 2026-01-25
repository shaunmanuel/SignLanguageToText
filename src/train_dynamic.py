import numpy as np
import os
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

DATA_DIR = "../data_dynamic"
SEQ_LEN = 30
FEATURES = 63

X = []
y = []

labels = sorted(os.listdir(DATA_DIR))

for label in labels:
    label_dir = os.path.join(DATA_DIR, label)
    for file in os.listdir(label_dir):
        seq = np.load(os.path.join(label_dir, file))
        if seq.shape == (SEQ_LEN, FEATURES):
            X.append(seq)
            y.append(label)

X = np.array(X)

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

np.save("../model/dynamic_labels.npy", encoder.classes_)
print("Dynamic labels:", encoder.classes_)

model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(SEQ_LEN, FEATURES)),
    LSTM(64),
    Dense(len(encoder.classes_), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(X, y_encoded, epochs=30, batch_size=8)

model.save("../model/dynamic_model.h5")
print("Dynamic model saved")

