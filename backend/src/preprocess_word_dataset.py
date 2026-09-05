import os
import json

import cv2
import numpy as np
import mediapipe as mp


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

INPUT_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "word_split"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "word_processed"
)

LABEL_PATH = os.path.join(
    OUTPUT_DIR,
    "labels.json"
)


# ============================================================
# SETTINGS
# ============================================================

SEQUENCE_LENGTH = 40

LANDMARKS_PER_HAND = 21
COORDINATES_PER_LANDMARK = 3

FEATURES_PER_HAND = (
    LANDMARKS_PER_HAND *
    COORDINATES_PER_LANDMARK
)

TOTAL_FEATURES = FEATURES_PER_HAND * 2


# ============================================================
# MEDIAPIPE
# ============================================================

mp_holistic = mp.solutions.holistic


# ============================================================
# HAND NORMALIZATION
# ============================================================

def normalize_hand(hand_landmarks):
    """
    Normalize one hand.

    21 landmarks × 3 coordinates = 63 features.

    Normalization:
    1. Wrist landmark (0) becomes the origin.
    2. Scale using landmark 9
       (middle finger MCP).
    """

    points = np.array(
        [
            [lm.x, lm.y, lm.z]
            for lm in hand_landmarks.landmark
        ],
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Wrist = landmark 0
    # --------------------------------------------------------

    points = points - points[0]

    # --------------------------------------------------------
    # Scale using middle finger MCP = landmark 9
    # --------------------------------------------------------

    scale = np.linalg.norm(points[9])

    if scale > 1e-6:
        points = points / scale

    return points.flatten().astype(np.float32)


# ============================================================
# FRAME FEATURE EXTRACTION
# ============================================================

def extract_frame_features(results):
    """
    Extract exactly 126 features from one frame.

    LEFT HAND first  = 63 features
    RIGHT HAND second = 63 features

    Total = 126 features.
    """

    left_hand = np.zeros(
        FEATURES_PER_HAND,
        dtype=np.float32
    )

    right_hand = np.zeros(
        FEATURES_PER_HAND,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # LEFT HAND
    # --------------------------------------------------------

    if results.left_hand_landmarks:

        left_hand = normalize_hand(
            results.left_hand_landmarks
        )

    # --------------------------------------------------------
    # RIGHT HAND
    # --------------------------------------------------------

    if results.right_hand_landmarks:

        right_hand = normalize_hand(
            results.right_hand_landmarks
        )

    # --------------------------------------------------------
    # IMPORTANT:
    # LEFT → RIGHT
    # Must match training and realtime inference.
    # --------------------------------------------------------

    return np.concatenate(
        [
            left_hand,
            right_hand
        ]
    ).astype(np.float32)


# ============================================================
# SELECT 40 FRAMES
# ============================================================

def select_frames(frames):
    """
    Select exactly 40 evenly distributed frames
    from the complete video.
    """

    if len(frames) == 0:
        return None

    indices = np.linspace(
        0,
        len(frames) - 1,
        SEQUENCE_LENGTH
    ).astype(int)

    selected = [
        frames[index]
        for index in indices
    ]

    return np.array(
        selected,
        dtype=np.float32
    )


# ============================================================
# PROCESS ONE VIDEO
# ============================================================

def process_video(video_path, output_path, holistic):
    """
    Process one video and save its landmark sequence.
    """

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return False

    frames = []

    while True:

        success, frame = cap.read()

        if not success:
            break

        # ----------------------------------------------------
        # BGR → RGB
        # ----------------------------------------------------

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # ----------------------------------------------------
        # MediaPipe
        # ----------------------------------------------------

        results = holistic.process(rgb)

        # ----------------------------------------------------
        # Extract 126 features
        # ----------------------------------------------------

        features = extract_frame_features(
            results
        )

        frames.append(features)

    cap.release()

    # --------------------------------------------------------
    # Make sure video contained frames
    # --------------------------------------------------------

    if len(frames) == 0:
        return False

    # --------------------------------------------------------
    # Select exactly 40 frames
    # --------------------------------------------------------

    sequence = select_frames(frames)

    if sequence is None:
        return False

    # --------------------------------------------------------
    # Verify shape
    # --------------------------------------------------------

    if sequence.shape != (
        SEQUENCE_LENGTH,
        TOTAL_FEATURES
    ):
        print(
            f"Unexpected shape: "
            f"{sequence.shape}"
        )

        return False

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    np.save(
        output_path,
        sequence
    )

    return True


# ============================================================
# PROCESS DATASET SPLIT
# ============================================================

def process_split(
    split_name,
    holistic
):

    input_split = os.path.join(
        INPUT_DIR,
        split_name
    )

    output_split = os.path.join(
        OUTPUT_DIR,
        split_name
    )

    if not os.path.exists(input_split):

        print(
            f"\nERROR: Missing folder: "
            f"{input_split}"
        )

        return 0, 0

    os.makedirs(
        output_split,
        exist_ok=True
    )

    successful = 0
    failed = 0

    print("\n" + "=" * 60)
    print(
        f"PROCESSING {split_name.upper()} SET"
    )
    print("=" * 60)

    classes = sorted(
        [
            name
            for name in os.listdir(input_split)
            if os.path.isdir(
                os.path.join(
                    input_split,
                    name
                )
            )
        ]
    )

    for class_name in classes:

        input_class = os.path.join(
            input_split,
            class_name
        )

        output_class = os.path.join(
            output_split,
            class_name
        )

        os.makedirs(
            output_class,
            exist_ok=True
        )

        videos = sorted(
            [
                file
                for file in os.listdir(input_class)
                if file.lower().endswith(
                    (".mp4", ".avi", ".mov", ".mkv")
                )
            ]
        )

        print(
            f"\n{class_name}: "
            f"{len(videos)} videos"
        )

        for video_index, video_name in enumerate(
            videos,
            start=1
        ):

            video_path = os.path.join(
                input_class,
                video_name
            )

            output_name = (
                os.path.splitext(video_name)[0]
                + ".npy"
            )

            output_path = os.path.join(
                output_class,
                output_name
            )

            try:

                success = process_video(
                    video_path,
                    output_path,
                    holistic
                )

                if success:
                    successful += 1

                else:
                    failed += 1

                print(
                    f"\r  Progress: "
                    f"{video_index}/{len(videos)}",
                    end=""
                )

            except Exception as e:

                failed += 1

                print(
                    f"\n  ERROR: "
                    f"{video_name}: {e}"
                )

        print()

    return successful, failed


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("ISL WORD DATASET PREPROCESSOR")
    print("=" * 60)

    print(
        f"\nInput directory:\n{INPUT_DIR}"
    )

    print(
        f"\nOutput directory:\n{OUTPUT_DIR}"
    )

    print(
        f"\nSequence length: "
        f"{SEQUENCE_LENGTH}"
    )

    print(
        f"Features per frame: "
        f"{TOTAL_FEATURES}"
    )

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Find labels from training set
    # --------------------------------------------------------

    train_dir = os.path.join(
        INPUT_DIR,
        "train"
    )

    if not os.path.exists(train_dir):

        print(
            "\nERROR: Training directory not found."
        )

        return

    labels = sorted(
        [
            name
            for name in os.listdir(train_dir)
            if os.path.isdir(
                os.path.join(
                    train_dir,
                    name
                )
            )
        ]
    )

    print("\nClasses:")

    for index, label in enumerate(labels):

        print(
            f"{index}: {label}"
        )

    # --------------------------------------------------------
    # Save labels
    # --------------------------------------------------------

    with open(
        LABEL_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            labels,
            f,
            indent=4
        )

    print(
        f"\nLabels saved to:\n{LABEL_PATH}"
    )

    # --------------------------------------------------------
    # MediaPipe Holistic
    # --------------------------------------------------------

    total_successful = 0
    total_failed = 0

    with mp_holistic.Holistic(

        static_image_mode=False,

        model_complexity=1,

        smooth_landmarks=True,

        enable_segmentation=False,

        refine_face_landmarks=False,

        min_detection_confidence=0.5,

        min_tracking_confidence=0.5

    ) as holistic:

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        successful, failed = process_split(
            "train",
            holistic
        )

        total_successful += successful
        total_failed += failed

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        successful, failed = process_split(
            "validation",
            holistic
        )

        total_successful += successful
        total_failed += failed

        # ----------------------------------------------------
        # Test
        # ----------------------------------------------------

        successful, failed = process_split(
            "test",
            holistic
        )

        total_successful += successful
        total_failed += failed

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)

    print(
        f"\nSuccessfully processed: "
        f"{total_successful}"
    )

    print(
        f"Failed: "
        f"{total_failed}"
    )

    print(
        f"Total: "
        f"{total_successful + total_failed}"
    )

    print(
        f"\nExpected shape for every file: "
        f"({SEQUENCE_LENGTH}, {TOTAL_FEATURES})"
    )

    print(
        f"\nProcessed data saved to:\n"
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()