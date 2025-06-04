import numpy as np
import pandas as pd
import mediapipe as mp
import joblib

from app.utils.alignment_check import estimate_yaw, estimate_roll, is_front_facing
from app.extract.extract_face_features import extract_features

MODEL_PATH = "app/models/face/face_model.joblib"
SCALER_PATH = "app/models/face/face_scaler.joblib"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
)

def predict_face_asymmetry(rgb_frame):
    h, w = rgb_frame.shape[:2]
    result = face_mesh.process(rgb_frame)

    if not result.multi_face_landmarks:
        return {
            "name": "face",
            "rate": 0.0,
            "Result": "failed to detect face"
        }

    landmarks = result.multi_face_landmarks[0].landmark
    yaw = estimate_yaw(landmarks, (w, h))
    roll = estimate_roll(landmarks)

    if not is_front_facing(yaw, roll):
        return {
            "name": "face",
            "rate": 0.0,
            "Result": "no front face"
        }
    else:
        features = extract_features(landmarks)
        features.extend([yaw, roll])
        features_arr = np.array(features).reshape(1, -1)

        columns = scaler.feature_names_in_
        features_df = pd.DataFrame(features_arr, columns=columns)
        features_scaled = scaler.transform(features_df)
        pred = model.predict(features_scaled)[0]
        percent = float(model.predict_proba(features_scaled)[0][1])

        return {
            "name": "face",
            "rate": round(percent, 4),
            "Result": "abnormal" if pred == 1 else "normal"
        }