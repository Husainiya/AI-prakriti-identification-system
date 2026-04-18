import pandas as pd

# Load labeled data
df = pd.read_csv("data/labeled_features.csv")

# Split classes
vata = df[df["label"] == "Vata"]
pitta = df[df["label"] == "Pitta"]
kapha = df[df["label"] == "Kapha"]

# Find smallest class
min_size = min(len(vata), len(pitta), len(kapha))

# Balance dataset
vata_bal = vata.sample(min_size, random_state=42)
pitta_bal = pitta.sample(min_size, random_state=42)
kapha_bal = kapha.sample(min_size, random_state=42)

# Combine
balanced_df = pd.concat([vata_bal, pitta_bal, kapha_bal])

# Shuffle
balanced_df = balanced_df.sample(frac=1, random_state=42)

# Save
balanced_df.to_csv("data/balanced_features.csv", index=False)

print("Balanced dataset created!")
print(balanced_df["label"].value_counts())