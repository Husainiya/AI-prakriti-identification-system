import cv2
import mediapipe as mp
import math
import joblib
import pandas as pd

# Questionnaire
from src.questionnaire.questions import questions
from src.questionnaire.scoring import adaptive_questionnaire

# -----------------------------
# LOAD MODEL
# -----------------------------
try:
    model = joblib.load("models/prakriti_model.pkl")
    print("Model loaded successfully!")
except:
    print("ERROR: Model not found!")
    exit()

# -----------------------------
# STEP 1: QUESTIONNAIRE
# -----------------------------
q_scores = adaptive_questionnaire(questions)
print("\nQuestionnaire Scores:", q_scores)

# -----------------------------
# STEP 2: LOAD IMAGE
# -----------------------------
image_path = input("\nEnter image path (e.g., data/test1.jpg): ")
image = cv2.imread(image_path)

if image is None:
    print("ERROR: Image not found!")
    exit()

image = cv2.resize(image, (500, 500))
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# -----------------------------
# STEP 3: FACE DETECTION
# -----------------------------
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)
result = face_mesh.process(rgb)

# -----------------------------
# STEP 4: FEATURE EXTRACTION
# -----------------------------
if result.multi_face_landmarks:
    for face_landmarks in result.multi_face_landmarks:

        h, w, _ = image.shape
        pts = face_landmarks.landmark

        def dist(p1, p2):
            return math.dist((p1.x, p1.y), (p2.x, p2.y))

        # -----------------------------
        # DRAW FACE MESH (IMPROVED STYLE)
        # -----------------------------
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing.DrawingSpec(
                color=(200, 200, 200), thickness=1
            )
        )

        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing.DrawingSpec(
                color=(0, 255, 0), thickness=2
            )
        )

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
        # NEW RATIOS
        # -----------------------------
        ratio_jaw = jaw_width / face_width if face_width != 0 else 0
        ratio_eye = eye_distance / face_width if face_width != 0 else 0
        ratio_forehead = forehead_height / face_height if face_height != 0 else 0

        # -----------------------------
        # SYMMETRY
        # -----------------------------
        symmetry = abs(pts[33].x - pts[263].x)

        # -----------------------------
        # NOSE & LIPS
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
        # HIGHLIGHT KEY POINTS (IMPROVED)
        # -----------------------------
        def get_point(p):
            return int(p.x * w), int(p.y * h)

        points = {
            "left": pts[234],
            "right": pts[454],
            "top": pts[10],
            "bottom": pts[152],
            "eye_l": pts[33],
            "eye_r": pts[263]
        }

        coords = {k: get_point(v) for k, v in points.items()}

        # Bigger points
        for p in coords.values():
            cv2.circle(image, p, 7, (0, 255, 255), -1)

        # Thicker lines
        cv2.line(image, coords["left"], coords["right"], (255, 0, 0), 3)
        cv2.line(image, coords["top"], coords["bottom"], (0, 0, 255), 3)
        cv2.line(image, coords["eye_l"], coords["eye_r"], (0, 255, 0), 3)

        # Labels
        cv2.putText(image, "Width", coords["left"], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0),1)
        cv2.putText(image, "Height", coords["top"], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),1)
        cv2.putText(image, "Eyes", coords["eye_l"], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0),1)

        # -----------------------------
        # CREATE FEATURE DATAFRAME
        # -----------------------------
        features = pd.DataFrame([[ 
            face_width, face_height, ratio,
            jaw_width, forehead_height, eye_distance,
            ratio_jaw, ratio_eye, ratio_forehead,
            symmetry, nose_width, lip_height,
            brightness, texture, hue, saturation, value
        ]], columns=[
            "face_width", "face_height", "ratio",
            "jaw_width", "forehead_height", "eye_distance",
            "ratio_jaw", "ratio_eye", "ratio_forehead",
            "symmetry", "nose_width", "lip_height",
            "brightness", "texture", "hue", "saturation", "value"
        ])

        # -----------------------------
        # ML PROBABILITY
        # -----------------------------
        probs = model.predict_proba(features)[0]
        classes = model.classes_

        ml_scores = {k: float(v) for k, v in zip(classes, probs)}

        print("\nML Probabilities:", ml_scores)

        # -----------------------------
        # FUSION
        # -----------------------------
        scores = {"Vata": 0.0, "Pitta": 0.0, "Kapha": 0.0}

        total_q = sum(q_scores.values())

        if total_q > 0:
            top_q = max(q_scores, key=q_scores.get)
            confidence_q = q_scores[top_q] / total_q
        else:
            confidence_q = 0

        print(f"Questionnaire Confidence: {round(confidence_q,2)}")

        # CONTINUOUS WEIGHTING
        confidence_q = max(0.4, min(confidence_q, 0.9))

        q_weight = 0.5 + (confidence_q - 0.4) * (0.4 / 0.5)
        ml_weight = 1 - q_weight

        print(f"Fusion Weights → Q: {round(q_weight,2)}, ML: {round(ml_weight,2)}")

        # APPLY QUESTIONNAIRE
        if total_q > 0:
            for d in scores:
                scores[d] += q_weight * (q_scores[d] / total_q)

        # APPLY ML
        for d in scores:
            scores[d] += ml_weight * ml_scores.get(d, 0)

        final_prakriti = max(scores, key=scores.get)

        print("\n--- FINAL RESULT ---")
        print("Scores:", {k: round(v, 3) for k, v in scores.items()})
        print("FINAL PRAKRITI:", final_prakriti)

        # DISPLAY RESULT
        cv2.putText(
            image,
            f"Prakriti: {final_prakriti}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

else:
    print("No face detected!")

# -----------------------------
# SHOW IMAGE
# -----------------------------
cv2.imshow("Prakriti System", image)
cv2.waitKey(0)
cv2.destroyAllWindows()