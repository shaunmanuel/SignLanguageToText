import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load dataset
data = pd.read_csv("../data/gestures.csv", header=None)

# Separate features and labels
X = data.iloc[:, :-1]   # 63 landmark values
y = data.iloc[:, -1]    # A, B, C

# Encode labels (A,B,C -> 0,1,2)
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Classes:", encoder.classes_)

