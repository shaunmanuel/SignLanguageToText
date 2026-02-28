import pandas as pd

# Load CSV
df = pd.read_csv("../data/gestures.csv", header=None)

# Clean labels (last column)
df.iloc[:, -1] = (
    df.iloc[:, -1]
    .astype(str)
    .str.strip()      # remove spaces
    .str.upper()      # force uppercase
)

# Save back
df.to_csv("../data/gestures.csv", index=False, header=False)

print("✅ Labels cleaned successfully")

