import cv2
import mediapipe as mp
import os
import pandas as pd
import math

# -----------------------------
# PATHS
# -----------------------------
DATASET_PATH = "data/raw/utkface"
OUTPUT_FILE = "data/features.csv"

# -----------------------------
# SETTINGS
# -----------------------------
MAX_IMAGES = 8000   # 🔥 Change between 5000–10000
SAVE_EVERY = 500    # Save progress every 500 images

# -----------------------------
# INITIALIZE MEDIAPIPE
# -----------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

data = []

print("Starting feature extraction...\n")

# -----------------------------
# PROCESS IMAGES
# -----------------------------
for i, filename in enumerate(os.listdir(DATASET_PATH)):

    if i >= MAX_IMAGES:
        break

    if not filename.lower().endswith(".jpg"):
        continue

    print(f"Processing {i}: {filename}")

    img_path = os.path.join(DATASET_PATH, filename)
    image = cv2.imread(img_path)

    if image is None:
        continue

    try:
        image = cv2.resize(image, (500, 500))
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        result = face_mesh.process(rgb)

        if not result.multi_face_landmarks:
            continue

        for face_landmarks in result.multi_face_landmarks:

            h, w, _ = image.shape
            pts = face_landmarks.landmark

            # -----------------------------
            # GEOMETRY FEATURES
            # -----------------------------
            left = pts[234]
            right = pts[454]
            face_width = math.dist((left.x, left.y), (right.x, right.y)) * w

            top = pts[10]
            bottom = pts[152]
            face_height = math.dist((top.x, top.y), (bottom.x, bottom.y)) * h

            ratio = face_width / face_height if face_height != 0 else 0

            jaw_l = pts[172]
            jaw_r = pts[397]
            jaw_width = math.dist((jaw_l.x, jaw_l.y), (jaw_r.x, jaw_r.y)) * w

            forehead_height = math.dist((top.x, top.y), (pts[168].x, pts[168].y)) * h

            # -----------------------------
            # EYE FEATURES
            # -----------------------------
            eye_l = pts[33]
            eye_r = pts[263]
            eye_distance = math.dist((eye_l.x, eye_l.y), (eye_r.x, eye_r.y)) * w

            # -----------------------------
            # SKIN FEATURES
            # -----------------------------
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            brightness = gray.mean()
            texture = gray.std()

            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hue = hsv[:, :, 0].mean()
            saturation = hsv[:, :, 1].mean()
            value = hsv[:, :, 2].mean()

            # -----------------------------
            # SAVE DATA
            # -----------------------------
            data.append([
                face_width,
                face_height,
                ratio,
                jaw_width,
                forehead_height,
                eye_distance,
                brightness,
                texture,
                hue,
                saturation,
                value
            ])

    except Exception as e:
        print("Error:", e)
        continue

    # -----------------------------
    # SAVE PARTIAL DATA (VERY IMPORTANT)
    # -----------------------------
    if i % SAVE_EVERY == 0 and len(data) > 0:
        df_temp = pd.DataFrame(data, columns=[
            "face_width",
            "face_height",
            "ratio",
            "jaw_width",
            "forehead_height",
            "eye_distance",
            "brightness",
            "texture",
            "hue",
            "saturation",
            "value"
        ])
        df_temp.to_csv(OUTPUT_FILE, index=False)
        print(f"Saved progress at {i} images")

# -----------------------------
# FINAL SAVE
# -----------------------------
df = pd.DataFrame(data, columns=[
    "face_width",
    "face_height",
    "ratio",
    "jaw_width",
    "forehead_height",
    "eye_distance",
    "brightness",
    "texture",
    "hue",
    "saturation",
    "value"
])

df.to_csv(OUTPUT_FILE, index=False)

print("\nFeature extraction completed!")
print("Total samples:", len(data))
print("Saved to:", OUTPUT_FILE)