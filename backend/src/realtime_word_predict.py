import os
import json
from collections import deque

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

MODEL_PATH = os.path.join(
    BASE_DIR, "models", "isl_word_model.keras"
)

LABEL_PATH = os.path.join(
    BASE_DIR, "models", "word_labels.json"
)


# ============================================================
# SETTINGS
# ============================================================

SEQUENCE_LENGTH = 40


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 60)
print("LOADING ISL WORD MODEL")
print("=" * 60)

model = tf.keras.models.load_model(MODEL_PATH)

with open(LABEL_PATH, "r", encoding="utf-8") as f:
    labels = json.load(f)

print("\nModel loaded successfully.")

print("\nClasses:")
for index, label in enumerate(labels):
    print(f"{index}: {label}")


# ============================================================
# MEDIAPIPE
# ============================================================

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


# ============================================================
# HAND NORMALIZATION
# ============================================================

def normalize_hand(hand_landmarks):
    """
    EXACTLY MATCHES THE DATASET PREPROCESSING.

    21 landmarks × 3 coordinates = 63 values.
    """

    points = np.array(
        [
            [lm.x, lm.y, lm.z]
            for lm in hand_landmarks.landmark
        ],
        dtype=np.float32
    )

    # Wrist = landmark 0
    wrist = points[0].copy()

    # Make wrist the origin
    points = points - wrist

    # Middle finger MCP = landmark 9
    scale = np.linalg.norm(points[9])

    if scale > 1e-6:
        points = points / scale

    return points.flatten()


# ============================================================
# FRAME FEATURE EXTRACTION
# ============================================================

def extract_frame_features(results):
    """
    EXACTLY MATCHES THE TRAINING PREPROCESSING.

    LEFT HAND first
    RIGHT HAND second

    63 + 63 = 126 features.
    """

    left_hand = np.zeros(
        63,
        dtype=np.float32
    )

    right_hand = np.zeros(
        63,
        dtype=np.float32
    )

    if results.left_hand_landmarks:
        left_hand = normalize_hand(
            results.left_hand_landmarks
        )

    if results.right_hand_landmarks:
        right_hand = normalize_hand(
            results.right_hand_landmarks
        )

    # IMPORTANT:
    # Training uses LEFT → RIGHT
    return np.concatenate(
        [
            left_hand,
            right_hand
        ]
    ).astype(np.float32)


# ============================================================
# TEXT TO SPEECH
# ============================================================

def speak(text):

    if text in [
        "Waiting...",
        "No sign detected",
        "Uncertain"
    ]:
        return

    try:

        import subprocess

        speech_text = text.replace(
            "_",
            " "
        )

        escaped_text = speech_text.replace(
            "'",
            "''"
        )

        command = (
            "Add-Type -AssemblyName System.Speech; "
            "$speak = New-Object "
            "System.Speech.Synthesis.SpeechSynthesizer; "
            f"$speak.Speak('{escaped_text}')"
        )

        subprocess.Popen(
            [
                "powershell",
                "-Command",
                command
            ]
        )

    except Exception as e:

        print(
            "\nSpeech error:",
            e
        )


# ============================================================
# START WEBCAM
# ============================================================

print("\n" + "=" * 60)
print("STARTING WEBCAM")
print("=" * 60)

print("\nControls:")
print("Q = Quit")
print("C = Clear sequence")
print("S = Speak prediction")


cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("\nERROR: Could not open webcam.")
    print("Check that your webcam is available.")

    raise SystemExit


# ============================================================
# SEQUENCE
# ============================================================

sequence = deque(
    maxlen=SEQUENCE_LENGTH
)

current_label = "No sign detected"
current_confidence = 0.0

top_predictions = [
    ("", 0.0),
    ("", 0.0),
    ("", 0.0)
]


# ============================================================
# MAIN LOOP
# ============================================================

with mp_holistic.Holistic(

    static_image_mode=False,

    model_complexity=1,

    smooth_landmarks=True,

    enable_segmentation=False,

    refine_face_landmarks=False,

    min_detection_confidence=0.5,

    min_tracking_confidence=0.5

) as holistic:

    while True:

        # ====================================================
        # READ FRAME
        # ====================================================

        success, frame = cap.read()

        if not success:

            print(
                "\nCould not read webcam frame."
            )

            break


        # ====================================================
        # MIRROR DISPLAY
        # ====================================================

        frame = cv2.flip(
            frame,
            1
        )


        # ====================================================
        # BGR → RGB
        # ====================================================

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        # ====================================================
        # MEDIAPIPE
        # ====================================================

        results = holistic.process(
            rgb
        )


        # ====================================================
        # DRAW LEFT HAND
        # ====================================================

        if results.left_hand_landmarks:

            mp_drawing.draw_landmarks(

                frame,

                results.left_hand_landmarks,

                mp_holistic.HAND_CONNECTIONS

            )


        # ====================================================
        # DRAW RIGHT HAND
        # ====================================================

        if results.right_hand_landmarks:

            mp_drawing.draw_landmarks(

                frame,

                results.right_hand_landmarks,

                mp_holistic.HAND_CONNECTIONS

            )


        # ====================================================
        # CHECK FOR HAND
        # ====================================================

        hand_detected = (

            results.left_hand_landmarks is not None

            or

            results.right_hand_landmarks is not None

        )


        # ====================================================
        # NO HAND DETECTED
        # ====================================================

        if not hand_detected:

            # IMPORTANT:
            # Do NOT feed empty frames to the classifier.

            sequence.clear()

            current_label = "No sign detected"
            current_confidence = 0.0

            top_predictions = [
                ("", 0.0),
                ("", 0.0),
                ("", 0.0)
            ]


        # ====================================================
        # HAND DETECTED
        # ====================================================

        else:

            features = extract_frame_features(
                results
            )

            sequence.append(
                features
            )


            # =================================================
            # PREDICT ONLY AFTER 40 REAL HAND FRAMES
            # =================================================

            if len(sequence) == SEQUENCE_LENGTH:

                input_data = np.expand_dims(

                    np.array(sequence),

                    axis=0

                )


                prediction = model.predict(

                    input_data,

                    verbose=0

                )[0]


                # ---------------------------------------------
                # TOP 3 PREDICTIONS
                # ---------------------------------------------

                top_indices = np.argsort(
                    prediction
                )[-3:][::-1]


                predicted_index = (
                    top_indices[0]
                )


                current_confidence = float(
                    prediction[predicted_index]
                )


                current_label = labels[
                    predicted_index
                ]


                top_predictions = [

                    (
                        labels[index],
                        float(prediction[index])
                    )

                    for index in top_indices

                ]


                # ---------------------------------------------
                # TERMINAL OUTPUT
                # ---------------------------------------------

                print(

                    f"\rTop predictions: "
                    f"{labels[top_indices[0]]} "
                    f"({prediction[top_indices[0]] * 100:.1f}%) | "
                    f"{labels[top_indices[1]]} "
                    f"({prediction[top_indices[1]] * 100:.1f}%) | "
                    f"{labels[top_indices[2]]} "
                    f"({prediction[top_indices[2]] * 100:.1f}%)",

                    end=""

                )


        # ====================================================
        # DISPLAY BOX
        # ====================================================

        cv2.rectangle(

            frame,

            (10, 10),

            (570, 205),

            (0, 0, 0),

            -1

        )


        # ====================================================
        # TITLE
        # ====================================================

        cv2.putText(

            frame,

            "ISL Word Recognition",

            (25, 40),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.8,

            (255, 255, 255),

            2

        )


        # ====================================================
        # PREDICTION
        # ====================================================

        cv2.putText(

            frame,

            f"Sign: {current_label}",

            (25, 75),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.8,

            (255, 255, 255),

            2

        )


        # ====================================================
        # CONFIDENCE
        # ====================================================

        cv2.putText(

            frame,

            f"Confidence: "
            f"{current_confidence * 100:.1f}%",

            (25, 105),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.65,

            (255, 255, 255),

            2

        )


        # ====================================================
        # FRAME COUNT
        # ====================================================

        cv2.putText(

            frame,

            f"Frames: "
            f"{len(sequence)}/{SEQUENCE_LENGTH}",

            (25, 135),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.6,

            (255, 255, 255),

            2

        )


        # ====================================================
        # TOP 3
        # ====================================================

        if hand_detected and len(sequence) == SEQUENCE_LENGTH:

            y_position = 165

            for label, confidence in top_predictions:

                text = (
                    f"{label}: "
                    f"{confidence * 100:.1f}%"
                )

                cv2.putText(

                    frame,

                    text,

                    (25, y_position),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.45,

                    (255, 255, 255),

                    1

                )

                y_position += 18


        # ====================================================
        # CONTROLS
        # ====================================================

        cv2.putText(

            frame,

            "Q: Quit | C: Clear | S: Speak",

            (25, frame.shape[0] - 20),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.55,

            (255, 255, 255),

            2

        )


        # ====================================================
        # SHOW CAMERA
        # ====================================================

        cv2.imshow(

            "Assistive Communication - ISL Word Recognition",

            frame

        )


        # ====================================================
        # KEYBOARD
        # ====================================================

        key = cv2.waitKey(1) & 0xFF


        # ----------------------------------------------------
        # QUIT
        # ----------------------------------------------------

        if key == ord("q"):

            break


        # ----------------------------------------------------
        # CLEAR
        # ----------------------------------------------------

        elif key == ord("c"):

            sequence.clear()

            current_label = "No sign detected"

            current_confidence = 0.0

            top_predictions = [
                ("", 0.0),
                ("", 0.0),
                ("", 0.0)
            ]

            print(
                "\n\nSequence cleared."
            )


        # ----------------------------------------------------
        # SPEAK
        # ----------------------------------------------------

        elif key == ord("s"):

            print(
                f"\nSpeaking: "
                f"{current_label}"
            )

            speak(
                current_label
            )


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

print("\n\nWebcam closed.")
print("Program finished.")