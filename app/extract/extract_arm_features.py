import cv2
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
pose = mp_pose.Pose()
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)

def extract_arm_features(landmarks, hand_directions):
    def to_np(idx):
        return np.array([landmarks[idx].x, landmarks[idx].y, landmarks[idx].z])
    l_upper = to_np(13) - to_np(11)
    r_upper = to_np(14) - to_np(12)
    def vector_angle(v1, v2):
        cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        return np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
    angle_diff = vector_angle(l_upper, r_upper)
    depth_diff = abs(to_np(15)[2] - to_np(16)[2])
    height_diff = abs(to_np(15)[1] - to_np(16)[1])
    hand_dir = np.mean(hand_directions) if hand_directions else 0.0
    return [angle_diff, depth_diff, height_diff, hand_dir]


def extract_arm_input(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
            return features + lh_finger_coords + rh_finger_coords
        except:
            return None
    return None

