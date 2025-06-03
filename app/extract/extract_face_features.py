import numpy as np

LEFT_EYE = 33
RIGHT_EYE = 263
NOSE_TIP = 1

symmetry_pairs = [(61, 291), (78, 308), (76, 306)]

def extract_features(landmarks):
    eye_dist = np.linalg.norm(
        np.array([landmarks[LEFT_EYE].x, landmarks[LEFT_EYE].y]) -
        np.array([landmarks[RIGHT_EYE].x, landmarks[RIGHT_EYE].y])
    )

    features = []

    for l_idx, r_idx in symmetry_pairs:
        lx, ly = landmarks[l_idx].x, landmarks[l_idx].y
        rx, ry = landmarks[r_idx].x, landmarks[r_idx].y

        dist = np.linalg.norm([lx - rx, ly - ry])
        dx = lx - rx
        dy = ly - ry

        nx_l = (lx - landmarks[NOSE_TIP].x) / (eye_dist + 1e-6)
        ny_l = (ly - landmarks[NOSE_TIP].y) / (eye_dist + 1e-6)
        nx_r = (rx - landmarks[NOSE_TIP].x) / (eye_dist + 1e-6)
        ny_r = (ry - landmarks[NOSE_TIP].y) / (eye_dist + 1e-6)

        features.extend([dist, dx, dy, nx_l, ny_l, nx_r, ny_r])

    return features

