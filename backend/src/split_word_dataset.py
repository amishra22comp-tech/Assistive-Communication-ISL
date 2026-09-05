from pathlib import Path
import random
import shutil

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent.parent

SOURCE_DIR = BASE_DIR / "dataset" / "word_selected"
SPLIT_DIR = BASE_DIR / "dataset" / "word_split"

TRAIN_DIR = SPLIT_DIR / "train"
VAL_DIR = SPLIT_DIR / "validation"
TEST_DIR = SPLIT_DIR / "test"


# ============================================================
# SETTINGS
# ============================================================

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


# ============================================================
# HELPER FUNCTION
# ============================================================

def get_videos(folder):
    """Return all supported video files inside a folder."""
    return [
        file
        for file in folder.iterdir()
        if file.is_file() and file.suffix.lower() in VIDEO_EXTENSIONS
    ]


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("WORD DATASET SPLITTER")
    print("=" * 60)

    if not SOURCE_DIR.exists():
        print(f"\nERROR: Source folder not found:")
        print(SOURCE_DIR)
        return

    # Make sure split directories exist
    for directory in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    random.seed(RANDOM_SEED)

    total_videos = 0

    # Get all word/class folders
    class_folders = sorted(
        [
            folder
            for folder in SOURCE_DIR.iterdir()
            if folder.is_dir()
        ]
    )

    if not class_folders:
        print("\nERROR: No word folders found.")
        return

    print(f"\nFound {len(class_folders)} word classes.\n")

    # ========================================================
    # PROCESS EACH CLASS
    # ========================================================

    for class_folder in class_folders:

        word = class_folder.name

        videos = get_videos(class_folder)

        if not videos:
            print(f"WARNING: {word} has no videos.")
            continue

        # Shuffle deterministically
        random.shuffle(videos)

        total = len(videos)

        # Calculate split sizes
        train_count = int(total * TRAIN_RATIO)
        val_count = int(total * VAL_RATIO)

        # Make sure at least one video goes into each split
        train_count = max(train_count, 1)
        val_count = max(val_count, 1)

        # Remaining videos go to test
        test_count = total - train_count - val_count

        # Safety check
        if test_count < 1:
            test_count = 1
            train_count -= 1

        # Split
        train_videos = videos[:train_count]

        val_videos = videos[
            train_count:train_count + val_count
        ]

        test_videos = videos[
            train_count + val_count:
        ]

        # Create class folders
        train_class_dir = TRAIN_DIR / word
        val_class_dir = VAL_DIR / word
        test_class_dir = TEST_DIR / word

        train_class_dir.mkdir(parents=True, exist_ok=True)
        val_class_dir.mkdir(parents=True, exist_ok=True)
        test_class_dir.mkdir(parents=True, exist_ok=True)

        # Copy files
        for video in train_videos:
            shutil.copy2(video, train_class_dir / video.name)

        for video in val_videos:
            shutil.copy2(video, val_class_dir / video.name)

        for video in test_videos:
            shutil.copy2(video, test_class_dir / video.name)

        total_videos += total

        print(
            f"{word:<15} "
            f"Total: {total:<3} | "
            f"Train: {len(train_videos):<3} | "
            f"Val: {len(val_videos):<3} | "
            f"Test: {len(test_videos):<3}"
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n" + "=" * 60)
    print("SPLIT COMPLETE")
    print("=" * 60)

    print(f"\nTotal videos processed: {total_videos}")

    print(f"\nTrain folder:")
    print(TRAIN_DIR)

    print(f"\nValidation folder:")
    print(VAL_DIR)

    print(f"\nTest folder:")
    print(TEST_DIR)

    print("\nThe original word_selected dataset was NOT modified.")


if __name__ == "__main__":
    main()