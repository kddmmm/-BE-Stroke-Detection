import numpy as np
import pandas as pd
import joblib
from collections import Counter
import time
import mediapipe as mp

from app.utils.webcam import WebcamManager
from app.extract.extract_arm_features import extract_arm_features

model = joblib.load("app/models/arm/arm_model.joblib")
scaler = joblib.load("app/models/arm/arm_scaler.joblib")

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
pose = mp_pose.Pose()
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)

def precheck_hand():
    for _ in range(10):  # 1초간 10프레임(0.1s마다)
        rgb = WebcamManager.instance().get_rgb_frame()
        if rgb is None:
            time.sleep(0.1)
            continue
        hands_result = hands.process(rgb)
        if hands_result and hands_result.multi_hand_landmarks:
            return True
        time.sleep(0.1)
    return False

def detect_arm():
    # 1초간 손 감지 전처리
    if not precheck_hand():
        return {
            "name": "error",
            "rate": 0.0,
            "result": "no hand detected"
        }

    results = []
    frame_count = 0
    max_frames = 50
    interval = 0.1

    while frame_count < max_frames:
        rgb = WebcamManager.instance().get_rgb_frame()
        if rgb is None:
            continue

        pose_result = pose.process(rgb)
        hands_result = hands.process(rgb)

        hand_dirs = []
        lh_finger_coords = [0.0] * 63
        rh_finger_coords = [0.0] * 63

        if hands_result.multi_hand_landmarks and hands_result.multi_handedness:
            for idx, hand_landmarks in enumerate(hands_result.multi_hand_landmarks):
                handedness = hands_result.multi_handedness[idx].classification[0].label
                wrist = hand_landmarks.landmark[0]
                index_tip = hand_landmarks.landmark[8]
                dx = index_tip.x - wrist.x
                dy = index_tip.y - wrist.y
                direction = np.arctan2(dy, dx)
                hand_dirs.append(direction)
                coords = [v for lm in hand_landmarks.landmark for v in (lm.x, lm.y, lm.z)]
                if handedness == 'Left':
                    lh_finger_coords = coords
                else:
                    rh_finger_coords = coords

        if pose_result.pose_landmarks:
            try:
                landmarks = pose_result.pose_landmarks.landmark
                features = extract_arm_features(landmarks, hand_dirs)
                input_features = features + lh_finger_coords + rh_finger_coords
                X_input = pd.DataFrame([input_features], columns=scaler.feature_names_in_)
                X_scaled = scaler.transform(X_input)
                pred = model.predict(X_scaled)[0]
                results.append(pred)
                frame_count += 1
            except:
                pass

        time.sleep(interval)

    count = Counter(results)
    normal_count = count[0]
    total = len(results)
    rate = (normal_count / total) if total > 0 else 0

    return {
        "result": "normal" if rate >= 0.8 else "abnormal",
        "rate": round(rate, 4)
    }
