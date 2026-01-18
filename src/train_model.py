import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import LabelEncoder

# ==============================
# LOAD DATA
# ==============================
data = pd.read_csv("../data/gestures.csv", header=None)

X = data.iloc[:, :-1]   # 63 landmark features
y = data.iloc[:, -1]    # labels (A, B, C)

# ==============================
# LABEL ENCODING
# ==============================
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# SAVE LABEL ORDER (CRITICAL FIX)
np.save("../model/labels.npy", encoder.classes_)

print("Label order saved:", encoder.classes_)

# ==============================
# BUILD MODEL
# ==============================
model = Sequential([
    Dense(128, activation="relu", input_shape=(X.shape[1],)),
    Dense(64, activation="relu"),
    Dense(len(encoder.classes_), activation="softmax")
])

# ==============================
# COMPILE MODEL
# ==============================
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ==============================
# TRAIN MODEL
# ==============================
model.fit(X, y_encoded, epochs=30, batch_size=16)

# ==============================
# SAVE MODEL
# ==============================
model.save("../model/gesture_model.h5")
print("Model trained and saved successfully")

