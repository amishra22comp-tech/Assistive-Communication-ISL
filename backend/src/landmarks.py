import numpy as np
import mediapipe as mp
from src.config import FEATURES_PER_FRAME

mp_hands = mp.solutions.hands

def create_hands():
    return mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

def extract_frame_landmarks(frame_bgr, hands):
    frame_rgb = frame_bgr[:, :, ::-1]
    results = hands.process(frame_rgb)

    left = np.zeros(63, dtype=np.float32)
    right = np.zeros(63, dtype=np.float32)

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(
            results.multi_hand_landmarks,
            results.multi_handedness,
        ):
            coords = []
            for lm in hand_landmarks.landmark:
                coords.extend([lm.x, lm.y, lm.z])

            label = handedness.classification[0].label
            if label == "Left":
                left = np.array(coords, dtype=np.float32)
            else:
                right = np.array(coords, dtype=np.float32)

    features = np.concatenate([left, right])
    assert features.shape[0] == FEATURES_PER_FRAME
    return features
