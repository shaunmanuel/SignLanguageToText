import pandas as pd

# Load your existing dataset
df_main = pd.read_csv("../data/gestures.csv", header=None)

# Load friend's dataset
df_friend = pd.read_csv("../data/friend_gestures.csv", header=None)

print("Before merge:")
print(df_main.iloc[:, -1].value_counts())

# Append friend's data
df_combined = pd.concat([df_main, df_friend], ignore_index=True)

print("\nAfter merge:")
print(df_combined.iloc[:, -1].value_counts())

# Save back to gestures.csv
df_combined.to_csv("../data/gestures.csv", index=False, header=False)

print("\n✅ Friend's data successfully appended to gestures.csv")
print("Total samples:", len(df_combined))

