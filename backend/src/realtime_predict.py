import json
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

from src.config import (
    MODEL_PATH,
    LABELS_PATH,
    SEQUENCE_LENGTH,
    FEATURES_PER_FRAME,
    MIN_CONFIDENCE,
)


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)


# ============================================================
# LOAD LABELS
# ============================================================

with open(
    LABELS_PATH,
    "r",
    encoding="utf-8",
) as f:
    labels = json.load(f)


print("Model loaded successfully.")
print("Labels:", labels)


# ============================================================
# MEDIAPIPE
# ============================================================

mp_holistic = mp.solutions.holistic


# ============================================================
# NORMALIZE HAND LANDMARKS
# Must match preprocessing exactly
# ============================================================

def normalize_hand(hand_landmarks):

    if hand_landmarks is None:
        return np.zeros(
            21 * 3,
            dtype=np.float32,
        )

    points = np.array(
        [
            [
                landmark.x,
                landmark.y,
                landmark.z,
            ]
            for landmark in hand_landmarks.landmark
        ],
        dtype=np.float32,
    )

    # Wrist is landmark 0
    wrist = points[0].copy()

    # Move wrist to origin
    points = points - wrist

    # Scale based on middle finger MCP
    scale = np.linalg.norm(
        points[9]
    )

    if scale > 1e-6:
        points = points / scale

    return points.flatten().astype(
        np.float32
    )


# ============================================================
# EXTRACT 126 FEATURES
# Left hand = 63
# Right hand = 63
# ============================================================

def extract_landmarks(results):

    left_hand = normalize_hand(
        results.left_hand_landmarks
    )

    right_hand = normalize_hand(
        results.right_hand_landmarks
    )

    landmarks = np.concatenate(
        [
            left_hand,
            right_hand,
        ]
    )

    return landmarks.astype(
        np.float32
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\nStarting webcam...")
    print("Press Q to quit.\n")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():

        print(
            "ERROR: Could not open webcam."
        )

        return


    # Store latest frames
    sequence = deque(
        maxlen=SEQUENCE_LENGTH
    )


    predicted_sentence = (
        "Collecting frames..."
    )

    confidence = 0.0


    with mp_holistic.Holistic(

        static_image_mode=False,

        model_complexity=1,

        smooth_landmarks=True,

        refine_face_landmarks=False,

        min_detection_confidence=0.5,

        min_tracking_confidence=0.5,

    ) as holistic:


        while True:


            # ====================================================
            # READ CAMERA
            # ====================================================

            success, frame = cap.read()

            if not success:

                print(
                    "Failed to read webcam frame."
                )

                break


            # IMPORTANT:
            # Do NOT flip before MediaPipe/model processing.
            # This keeps hand orientation consistent with training.
            frame_rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            frame_rgb.flags.writeable = False


            # ====================================================
            # MEDIAPIPE DETECTION
            # ====================================================

            results = holistic.process(
                frame_rgb
            )


            # ====================================================
            # EXTRACT LANDMARKS
            # ====================================================

            landmarks = extract_landmarks(
                results
            )

            sequence.append(
                landmarks
            )


            # ====================================================
            # MODEL PREDICTION
            # ====================================================

            if len(sequence) == SEQUENCE_LENGTH:


                input_sequence = np.array(
                    sequence,
                    dtype=np.float32,
                )


                probabilities = model.predict(

                    input_sequence[
                        np.newaxis,
                        ...
                    ],

                    verbose=0,

                )[0]


                predicted_index = int(

                    np.argmax(
                        probabilities
                    )

                )


                confidence = float(

                    probabilities[
                        predicted_index
                    ]

                )


                predicted_label = labels[
                    predicted_index
                ]


                if confidence >= MIN_CONFIDENCE:

                    predicted_sentence = (
                        predicted_label
                    )

                else:

                    predicted_sentence = (
                        "Low confidence"
                    )


            # ====================================================
            # FLIP ONLY FOR DISPLAY
            # ====================================================

            display_frame = cv2.flip(
                frame,
                1,
            )


            # ====================================================
            # DISPLAY TEXT
            # ====================================================

            display_text = (
                f"{predicted_sentence}"
            )


            confidence_text = (
                f"Confidence: "
                f"{confidence:.2%}"
            )


            frame_count_text = (
                f"Frames: "
                f"{len(sequence)}/"
                f"{SEQUENCE_LENGTH}"
            )


            cv2.putText(

                display_frame,

                display_text,

                (30, 50),

                cv2.FONT_HERSHEY_SIMPLEX,

                1,

                (0, 255, 0),

                2,

                cv2.LINE_AA,

            )


            cv2.putText(

                display_frame,

                confidence_text,

                (30, 90),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.7,

                (255, 255, 255),

                2,

                cv2.LINE_AA,

            )


            cv2.putText(

                display_frame,

                frame_count_text,

                (30, 130),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.7,

                (255, 255, 255),

                2,

                cv2.LINE_AA,

            )


            # ====================================================
            # SHOW WINDOW
            # ====================================================

            cv2.imshow(

                "ISL Sentence Recognition",

                display_frame,

            )


            # ====================================================
            # QUIT
            # ====================================================

            key = cv2.waitKey(1) & 0xFF


            if key == ord("q"):

                break


    # ============================================================
    # CLEANUP
    # ============================================================

    cap.release()

    cv2.destroyAllWindows()

    print("\nWebcam closed.")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()