import pandas as pd

df = pd.read_csv("../data/gestures.csv", header=None)

# Clean label column
df.iloc[:, -1] = (
    df.iloc[:, -1]
    .astype(str)
    .str.strip()      # remove spaces
    .str.upper()      # convert to uppercase
)

df.to_csv("../data/gestures.csv", index=False, header=False)

print("✅ Labels cleaned successfully")
print(df.iloc[:, -1].value_counts())

