import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# ==============================
# LOAD DATA (UNCHANGED)
# ==============================
data = pd.read_csv("../data/gestures.csv", header=None)

X = data.iloc[:, :-1].values   # 63 raw landmark features
y = data.iloc[:, -1].values

# ==============================
# LABEL ENCODING
# ==============================
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Save labels
np.save("../model/labels.npy", encoder.classes_)
print("✅ Labels saved:", encoder.classes_)

num_classes = len(encoder.classes_)

# ==============================
# MODEL (MAX ACCURACY WITHOUT FEATURE CHANGE)
# ==============================
model = Sequential([
    Dense(512, activation="relu", input_shape=(63,)),
    Dense(256, activation="relu"),
    Dense(128, activation="relu"),
    Dense(64, activation="relu"),
    Dense(num_classes, activation="softmax")
])

# ==============================
# COMPILE
# ==============================
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ==============================
# EARLY STOPPING
# ==============================
early_stop = EarlyStopping(
    monitor="loss",
    patience=7,
    restore_best_weights=True
)

# ==============================
# TRAIN
# ==============================
model.fit(
    X,
    y_encoded,
    epochs=60,
    batch_size=32,
    callbacks=[early_stop]
)

# ==============================
# SAVE MODEL
# ==============================
model.save("../model/gesture_model.h5")
print("✅ High-accuracy model trained without changing dataset")

