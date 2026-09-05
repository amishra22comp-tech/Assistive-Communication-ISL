import cv2
import numpy as np

from src.config import (
    RAW_DATASET_DIR,
    PROCESSED_DATASET_DIR,
    TARGET_SENTENCES,
)
from src.landmarks import create_hands, extract_frame_landmarks
from src.preprocess import prepare_sequence

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def get_videos_for_sentence(sentence: str):
    sentence_dir = RAW_DATASET_DIR / sentence

    if not sentence_dir.exists():
        print(f"WARNING: Folder not found: {sentence_dir}")
        return []

    return sorted(
        [
            path
            for path in sentence_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in VIDEO_EXTENSIONS
        ]
    )


def extract_video(video_path, hands):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    sequence = []

    while True:
        ok, frame = cap.read()

        if not ok:
            break

        features = extract_frame_landmarks(frame, hands)
        sequence.append(features)

    cap.release()

    if not sequence:
        raise RuntimeError(f"No frames extracted from: {video_path}")

    return prepare_sequence(sequence)


def main():
    if not RAW_DATASET_DIR.exists():
        print("ERROR: dataset/raw does not exist.")
        return

    PROCESSED_DATASET_DIR.mkdir(parents=True, exist_ok=True)

    total_processed = 0

    with create_hands() as hands:
        for sentence in TARGET_SENTENCES:
            videos = get_videos_for_sentence(sentence)

            output_dir = PROCESSED_DATASET_DIR / sentence
            output_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n{sentence}: {len(videos)} video(s)")

            for index, video_path in enumerate(videos):
                output_path = output_dir / f"{index:04d}.npy"

                try:
                    print(f"Processing: {video_path.name}")

                    sequence = extract_video(video_path, hands)

                    np.save(output_path, sequence)

                    print(
                        f"Saved: {output_path.name} "
                        f"| Shape: {sequence.shape}"
                    )

                    total_processed += 1

                except Exception as exc:
                    print(f"FAILED: {video_path.name}")
                    print(f"Reason: {exc}")

    print("\n" + "=" * 50)
    print("LANDMARK EXTRACTION COMPLETE")
    print(f"Total videos processed: {total_processed}")
    print("=" * 50)


if __name__ == "__main__":
    main()