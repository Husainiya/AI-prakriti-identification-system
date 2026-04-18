import cv2
import mediapipe as mp
import math
import joblib
import numpy as np
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
image = cv2.imread("data/sample.png")

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

        # -----------------------------
        # DRAW FACE MESH
        # -----------------------------
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style()
        )

        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_styles.get_default_face_mesh_contours_style()
        )

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
        # HIGHLIGHT KEY POINTS
        # -----------------------------
        def get_point(p):
            return int(p.x * w), int(p.y * h)

        left_pt = get_point(pts[234])
        right_pt = get_point(pts[454])
        top_pt = get_point(pts[10])
        bottom_pt = get_point(pts[152])
        eye_l_pt = get_point(pts[33])
        eye_r_pt = get_point(pts[263])

        for p in [left_pt, right_pt, top_pt, bottom_pt, eye_l_pt, eye_r_pt]:
            cv2.circle(image, p, 5, (0, 255, 255), -1)

        cv2.line(image, left_pt, right_pt, (255, 0, 0), 2)
        cv2.line(image, top_pt, bottom_pt, (0, 0, 255), 2)
        cv2.line(image, eye_l_pt, eye_r_pt, (0, 255, 0), 2)

        # -----------------------------
        # CREATE FEATURE DATAFRAME
        # -----------------------------
        features = pd.DataFrame([[
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
        ]], columns=[
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

        # -----------------------------
        # ML PREDICTION
        # -----------------------------
        ml_prediction = model.predict(features)[0]
        print("\nML Prediction:", ml_prediction)

        # -----------------------------
        # DYNAMIC FUSION
        # -----------------------------
        scores = {"Vata": 0.0, "Pitta": 0.0, "Kapha": 0.0}

        total_q = sum(q_scores.values())

        # Calculate confidence
        if total_q > 0:
            top_q = max(q_scores, key=q_scores.get)
            confidence_q = q_scores[top_q] / total_q
        else:
            confidence_q = 0

        print(f"Questionnaire Confidence: {round(confidence_q,2)}")

        # Dynamic weights
        if confidence_q >= 0.75:
            q_weight = 0.85
            ml_weight = 0.15
        elif confidence_q >= 0.60:
            q_weight = 0.70
            ml_weight = 0.30
        else:
            q_weight = 0.55
            ml_weight = 0.45

        print(f"Fusion Weights → Q: {q_weight}, ML: {ml_weight}")

        # Apply questionnaire
        if total_q > 0:
            for d in scores:
                scores[d] += q_weight * (q_scores[d] / total_q)

        # Apply ML
        scores[ml_prediction] += ml_weight

        final_prakriti = max(scores, key=scores.get)

        print("\n--- FINAL RESULT ---")
        print("FINAL PRAKRITI:", final_prakriti)

        # Display
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