import cv2
import numpy as np
from tqdm import tqdm
import mediapipe as mp

from src.config import (
    RAW_DATASET_DIR,
    PROCESSED_DATASET_DIR,
    TARGET_SENTENCES,
    SEQUENCE_LENGTH,
    FEATURES_PER_FRAME,
)


mp_holistic = mp.solutions.holistic


def normalize_hand(hand_landmarks):
    """
    Normalize hand landmarks relative to the wrist.
    """

    if hand_landmarks is None:
        return np.zeros(21 * 3, dtype=np.float32)

    points = np.array(
        [
            [landmark.x, landmark.y, landmark.z]
            for landmark in hand_landmarks.landmark
        ],
        dtype=np.float32,
    )

    # Move wrist to origin
    wrist = points[0].copy()
    points = points - wrist

    # Scale relative to hand size
    scale = np.linalg.norm(points[9])

    if scale > 1e-6:
        points = points / scale

    return points.flatten().astype(np.float32)


def normalize_pose(pose_landmarks):
    """
    Normalize full-body pose landmarks relative to the shoulders.

    This preserves body motion and makes the landmarks
    less dependent on camera position and body size.
    """

    if pose_landmarks is None:
        return np.zeros(33 * 3, dtype=np.float32)

    points = np.array(
        [
            [landmark.x, landmark.y, landmark.z]
            for landmark in pose_landmarks.landmark
        ],
        dtype=np.float32,
    )

    # MediaPipe pose landmark indices:
    # 11 = left shoulder
    # 12 = right shoulder

    left_shoulder = points[11]
    right_shoulder = points[12]

    # Use midpoint between shoulders as origin
    shoulder_center = (
        left_shoulder + right_shoulder
    ) / 2.0

    points = points - shoulder_center

    # Normalize by shoulder width
    shoulder_width = np.linalg.norm(
        left_shoulder - right_shoulder
    )

    if shoulder_width > 1e-6:
        points = points / shoulder_width

    return points.flatten().astype(np.float32)


def extract_landmarks(results):
    """
    Feature vector per frame:

    Left hand:  21 x 3 = 63
    Right hand: 21 x 3 = 63
    Pose:       33 x 3 = 99

    Total: 225 features
    """

    left_hand = normalize_hand(
        results.left_hand_landmarks
    )

    right_hand = normalize_hand(
        results.right_hand_landmarks
    )

    pose = normalize_pose(
        results.pose_landmarks
    )

    landmarks = np.concatenate(
        [
            left_hand,
            right_hand,
            pose,
        ]
    )

    return landmarks.astype(np.float32)


def select_frames(total_frames):
    """
    Select SEQUENCE_LENGTH frames evenly
    across the complete video.
    """

    if total_frames <= 0:
        return []

    indices = np.linspace(
        0,
        total_frames - 1,
        SEQUENCE_LENGTH,
    )

    return indices.astype(int)


def process_video(video_path):
    """
    Convert one video into:

    (
        SEQUENCE_LENGTH,
        FEATURES_PER_FRAME
    )
    """

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        print(
            f"\nCould not open: {video_path}"
        )

        return None

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if total_frames <= 0:

        cap.release()

        print(
            f"\nNo frames found: {video_path}"
        )

        return None

    frame_indices = select_frames(
        total_frames
    )

    sequence = []

    with mp_holistic.Holistic(

        static_image_mode=False,

        model_complexity=1,

        smooth_landmarks=True,

        refine_face_landmarks=False,

        min_detection_confidence=0.5,

        min_tracking_confidence=0.5,

    ) as holistic:

        for frame_index in frame_indices:

            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                int(frame_index),
            )

            success, frame = cap.read()

            if not success:

                sequence.append(
                    np.zeros(
                        FEATURES_PER_FRAME,
                        dtype=np.float32,
                    )
                )

                continue

            frame_rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            frame_rgb.flags.writeable = False

            results = holistic.process(
                frame_rgb
            )

            landmarks = extract_landmarks(
                results
            )

            sequence.append(
                landmarks
            )

    cap.release()

    sequence = np.asarray(
        sequence,
        dtype=np.float32,
    )

    expected_shape = (
        SEQUENCE_LENGTH,
        FEATURES_PER_FRAME,
    )

    if sequence.shape != expected_shape:

        print(
            f"\nBad output shape: "
            f"{video_path} "
            f"{sequence.shape}"
        )

        print(
            f"Expected: "
            f"{expected_shape}"
        )

        return None

    return sequence


def main():

    print(
        "\nStarting preprocessing..."
    )

    print(
        "\nRaw dataset:"
    )

    print(
        RAW_DATASET_DIR
    )

    print(
        "\nProcessed dataset:"
    )

    print(
        PROCESSED_DATASET_DIR
    )

    PROCESSED_DATASET_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    total_processed = 0
    total_failed = 0

    for sentence in TARGET_SENTENCES:

        source_dir = (
            RAW_DATASET_DIR /
            sentence
        )

        output_dir = (
            PROCESSED_DATASET_DIR /
            sentence
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        if not source_dir.exists():

            print(
                f"\nWARNING: "
                f"Folder missing: "
                f"{source_dir}"
            )

            continue

        video_files = sorted(
            [
                path
                for path in source_dir.iterdir()
                if (
                    path.is_file()
                    and path.suffix.lower()
                    in (
                        ".mp4",
                        ".avi",
                        ".mov",
                        ".mkv",
                    )
                )
            ]
        )

        print(
            f"\n{sentence}: "
            f"{len(video_files)} videos"
        )

        for index, video_path in enumerate(

            tqdm(
                video_files,
                desc=sentence,
            )
        ):

            sequence = process_video(
                video_path
            )

            if sequence is None:

                total_failed += 1

                continue

            output_path = (
                output_dir /
                f"{index:04d}.npy"
            )

            np.save(
                output_path,
                sequence,
            )

            total_processed += 1

    print(
        "\n" + "=" * 50
    )

    print(
        "PREPROCESSING COMPLETE"
    )

    print(
        "=" * 50
    )

    print(
        f"Successfully processed: "
        f"{total_processed}"
    )

    print(
        f"Failed videos: "
        f"{total_failed}"
    )

    print(
        f"Output location: "
        f"{PROCESSED_DATASET_DIR}"
    )


if __name__ == "__main__":
    main()