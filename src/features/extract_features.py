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

MAX_IMAGES = 8000
SAVE_EVERY = 500

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

data = []

print("Starting feature extraction...\n")

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

            def dist(p1, p2):
                return math.dist((p1.x, p1.y), (p2.x, p2.y))

            # -----------------------------
            # BASIC GEOMETRY
            # -----------------------------
            face_width = dist(pts[234], pts[454]) * w
            face_height = dist(pts[10], pts[152]) * h
            ratio = face_width / face_height if face_height != 0 else 0

            jaw_width = dist(pts[172], pts[397]) * w
            forehead_height = dist(pts[10], pts[168]) * h
            eye_distance = dist(pts[33], pts[263]) * w

            # -----------------------------
            # NEW RATIOS (IMPORTANT)
            # -----------------------------
            ratio_jaw = jaw_width / face_width if face_width != 0 else 0
            ratio_eye = eye_distance / face_width if face_width != 0 else 0
            ratio_forehead = forehead_height / face_height if face_height != 0 else 0

            # -----------------------------
            # SYMMETRY
            # -----------------------------
            symmetry = abs(pts[33].x - pts[263].x)

            # -----------------------------
            # NOSE & LIP
            # -----------------------------
            nose_width = dist(pts[97], pts[326]) * w
            lip_height = dist(pts[13], pts[14]) * h

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
            # SAVE
            # -----------------------------
            data.append([
                face_width, face_height, ratio,
                jaw_width, forehead_height, eye_distance,
                ratio_jaw, ratio_eye, ratio_forehead,
                symmetry, nose_width, lip_height,
                brightness, texture, hue, saturation, value
            ])

    except Exception as e:
        print("Error:", e)
        continue

    if i % SAVE_EVERY == 0 and len(data) > 0:
        df_temp = pd.DataFrame(data, columns=[
            "face_width", "face_height", "ratio",
            "jaw_width", "forehead_height", "eye_distance",
            "ratio_jaw", "ratio_eye", "ratio_forehead",
            "symmetry", "nose_width", "lip_height",
            "brightness", "texture", "hue", "saturation", "value"
        ])
        df_temp.to_csv(OUTPUT_FILE, index=False)
        print(f"Saved progress at {i}")

# FINAL SAVE
df = pd.DataFrame(data, columns=[
    "face_width", "face_height", "ratio",
    "jaw_width", "forehead_height", "eye_distance",
    "ratio_jaw", "ratio_eye", "ratio_forehead",
    "symmetry", "nose_width", "lip_height",
    "brightness", "texture", "hue", "saturation", "value"
])

df.to_csv(OUTPUT_FILE, index=False)

print("\nFeature extraction completed!")
print("Total samples:", len(data))