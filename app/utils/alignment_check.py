import cv2
import numpy as np

LEFT_EYE = 33
RIGHT_EYE = 263
NOSE_TIP = 1

def estimate_yaw(landmarks, image_size):
    image_w, image_h = image_size
    indices = [1, 33, 263, 61, 291, 199]
    model_points = np.array([
        [0.0, 0.0, 0.0],
        [-30.0, -30.0, -30.0],
        [30.0, -30.0, -30.0],
        [-30.0, 30.0, -30.0],
        [30.0, 30.0, -30.0],
        [0.0, 50.0, -10.0]
    ], dtype=np.float32)

    image_points = [(int(landmarks[idx].x * image_w), int(landmarks[idx].y * image_h)) for idx in indices]
    image_points = np.array(image_points, dtype=np.float32)

    focal_length = image_w
    center = (image_w / 2, image_h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")
    dist_coeffs = np.zeros((4, 1))

    success, rotation_vector, _ = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return 0.0

    rotation_mat, _ = cv2.Rodrigues(rotation_vector)
    sy = np.sqrt(rotation_mat[0, 0] ** 2 + rotation_mat[1, 0] ** 2)
    yaw = np.degrees(np.arctan2(-rotation_mat[2, 0], sy))
    return yaw

def estimate_roll(landmarks):
    dx = landmarks[RIGHT_EYE].x - landmarks[LEFT_EYE].x
    dy = landmarks[RIGHT_EYE].y - landmarks[LEFT_EYE].y
    return np.degrees(np.arctan2(dy, dx))

def is_front_facing(yaw, roll, yaw_thresh=10, roll_thresh=14):
    return abs(yaw) <= yaw_thresh and abs(roll) <= roll_thresh
