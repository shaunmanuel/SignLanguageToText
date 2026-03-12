import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# ==============================
# LOAD DATA
# ==============================
data = pd.read_csv("../data/gestures.csv", header=None)

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# convert labels to string
y = y.astype(str)

# ==============================
# LABEL ENCODING
# ==============================
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

np.save("../model/labels.npy", encoder.classes_)
print("Label order saved:", encoder.classes_)

# ==============================
# FEATURE SCALING
# ==============================
scaler = StandardScaler()
X = scaler.fit_transform(X)

# save scaler (important for prediction)
import pickle
pickle.dump(scaler, open("../model/scaler.pkl", "wb"))

# ==============================
# TRAIN TEST SPLIT
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# ==============================
# BUILD MODEL
# ==============================
model = Sequential([
    Dense(256, activation="relu", input_shape=(X.shape[1],)),
    BatchNormalization(),
    Dropout(0.3),

    Dense(128, activation="relu"),
    BatchNormalization(),
    Dropout(0.3),

    Dense(64, activation="relu"),

    Dense(len(encoder.classes_), activation="softmax")
])

# ==============================
# COMPILE
# ==============================
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ==============================
# TRAIN
# ==============================
model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# ==============================
# TEST ACCURACY
# ==============================
loss, acc = model.evaluate(X_test, y_test)
print("Test Accuracy:", acc)

# ==============================
# SAVE MODEL
# ==============================
model.save("../model/gesture_model.h5")

print("Model trained and saved successfully")

