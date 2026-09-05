from pathlib import Path


# ============================================================
# BASE DIRECTORY
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent


# ============================================================
# DATASET DIRECTORIES
# ============================================================

RAW_DATASET_DIR = BASE_DIR / "dataset" / "raw"

PROCESSED_DATASET_DIR = (
    BASE_DIR / "dataset" / "processed"
)


# ============================================================
# MODEL DIRECTORIES AND FILES
# ============================================================

MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = (
    MODEL_DIR / "isl_sentence_model.keras"
)

LABELS_PATH = (
    MODEL_DIR / "labels.json"
)


# ============================================================
# INDIAN SIGN LANGUAGE SENTENCES
# ============================================================

TARGET_SENTENCES = [
    "are you free today",
    "can i help you",
    "had your food",
    "help me",
    "i am fine. thank you sir",
    "i am hungry",
    "i am tired",
    "i am very happy",
    "i need water",
    "nice to meet you",
    "take care of yourself",
    "what happened",
    "why are you crying",
    "you are welcome",
]


# ============================================================
# VIDEO / SEQUENCE SETTINGS
# ============================================================

SEQUENCE_LENGTH = 40


# ============================================================
# MEDIAPIPE LANDMARK SETTINGS
# ============================================================

LANDMARKS_PER_HAND = 21
LANDMARKS_PER_POSE = 33

COORDINATES_PER_LANDMARK = 3


FEATURES_PER_HAND = (
    LANDMARKS_PER_HAND *
    COORDINATES_PER_LANDMARK
)

FEATURES_PER_POSE = (
    LANDMARKS_PER_POSE *
    COORDINATES_PER_LANDMARK
)


# Left hand = 63
# Right hand = 63
# Full body pose = 99
#
# Total = 225 features per frame

FEATURES_PER_FRAME = (
    FEATURES_PER_HAND +
    FEATURES_PER_HAND +
    FEATURES_PER_POSE
)


# ============================================================
# MEDIAPIPE DETECTION SETTINGS
# ============================================================

MIN_CONFIDENCE = 0.70