import pandas as pd

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("data/features.csv")

# -----------------------------
# NORMALIZATION (IMPORTANT)
# -----------------------------
# Scale features to 0–1 range (helps fairness)
features_to_normalize = [
    "ratio", "brightness", "texture", "saturation"
]

for col in features_to_normalize:
    df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

# -----------------------------
# LABELING FUNCTION
# -----------------------------
def assign_prakriti(row):

    scores = {"Vata": 0, "Pitta": 0, "Kapha": 0}

    ratio = row["ratio"]
    brightness = row["brightness"]
    texture = row["texture"]
    saturation = row["saturation"]

    # -----------------------------
    # FACE SHAPE
    # -----------------------------
    if ratio < 0.4:
        scores["Vata"] += 2
    elif ratio < 0.6:
        scores["Pitta"] += 2
    else:
        scores["Kapha"] += 2

    # -----------------------------
    # SKIN BRIGHTNESS
    # -----------------------------
    if brightness < 0.3:
        scores["Vata"] += 1
    elif brightness < 0.6:
        scores["Pitta"] += 1
    else:
        scores["Kapha"] += 1

    # -----------------------------
    # TEXTURE
    # -----------------------------
    if texture > 0.7:
        scores["Vata"] += 1
    elif texture > 0.4:
        scores["Pitta"] += 1
    else:
        scores["Kapha"] += 1

    # -----------------------------
    # SATURATION
    # -----------------------------
    if saturation > 0.7:
        scores["Pitta"] += 1
    elif saturation > 0.4:
        scores["Kapha"] += 1
    else:
        scores["Vata"] += 1

    return max(scores, key=scores.get)


# -----------------------------
# APPLY LABELING
# -----------------------------
df["label"] = df.apply(assign_prakriti, axis=1)

# -----------------------------
# SAVE
# -----------------------------
df.to_csv("data/labeled_features.csv", index=False)

print("\n Labels generated successfully!")
print("\nLabel distribution:\n")
print(df["label"].value_counts())