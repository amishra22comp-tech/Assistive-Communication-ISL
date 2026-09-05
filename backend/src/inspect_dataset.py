from pathlib import Path
from src.config import RAW_DATASET_DIR

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

def main():
    print(f"Inspecting: {RAW_DATASET_DIR.resolve()}")
    if not RAW_DATASET_DIR.exists():
        print("ERROR: dataset/raw does not exist.")
        return

    folders = [p for p in RAW_DATASET_DIR.rglob("*") if p.is_dir()]
    videos = [p for p in RAW_DATASET_DIR.rglob("*") if p.suffix.lower() in VIDEO_EXTENSIONS]

    print(f"Folders: {len(folders)}")
    print(f"Videos: {len(videos)}")
    print("\nTop-level contents:")
    for item in RAW_DATASET_DIR.iterdir():
        print(" -", item.name)

    print("\nFirst 30 video paths:")
    for video in videos[:30]:
        print(video.relative_to(RAW_DATASET_DIR))

    print("\nTIP: inspect these paths before changing TARGET_SENTENCES in config.py.")

if __name__ == "__main__":
    main()
