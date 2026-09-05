from pathlib import Path
import json

import numpy as np
import tensorflow as tf


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent.parent

PROCESSED_DIR = BASE_DIR / "dataset" / "word_processed"

TRAIN_DIR = PROCESSED_DIR / "train"
VAL_DIR = PROCESSED_DIR / "validation"
TEST_DIR = PROCESSED_DIR / "test"

MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "isl_word_model.keras"
LABELS_PATH = MODEL_DIR / "word_labels.json"


# ============================================================
# SETTINGS
# ============================================================

SEQUENCE_LENGTH = 40
FEATURES_PER_FRAME = 126

EPOCHS = 100
BATCH_SIZE = 8

RANDOM_SEED = 42


# ============================================================
# LOAD SPLIT
# ============================================================

def load_split(split_dir, label_to_index):
    """
    Load all .npy files from a dataset split.

    Expected structure:

        split_dir/
            word1/
                video1.npy
                video2.npy
            word2/
                video1.npy
                ...

    Returns:
        X, y
    """

    X = []
    y = []

    print(f"\nLoading: {split_dir}")

    class_dirs = sorted(
        [
            folder
            for folder in split_dir.iterdir()
            if folder.is_dir()
        ]
    )

    for class_dir in class_dirs:

        word = class_dir.name

        if word not in label_to_index:
            print(f"WARNING: Unknown class: {word}")
            continue

        files = sorted(class_dir.glob("*.npy"))

        print(
            f"{word:<15} : {len(files)} samples"
        )

        for file in files:

            try:

                sequence = np.load(file)

                expected_shape = (
                    SEQUENCE_LENGTH,
                    FEATURES_PER_FRAME
                )

                if sequence.shape != expected_shape:

                    print(
                        f"Skipping {file.name}: "
                        f"expected {expected_shape}, "
                        f"found {sequence.shape}"
                    )

                    continue

                X.append(
                    sequence.astype(np.float32)
                )

                y.append(
                    label_to_index[word]
                )

            except Exception as e:

                print(
                    f"Failed to load {file}: {e}"
                )

    if not X:
        raise RuntimeError(
            f"No valid samples found in {split_dir}"
        )

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.int32)
    )


# ============================================================
# AUGMENTATION
# ============================================================

def temporal_shift(sequence, rng):

    shift = int(
        rng.integers(-2, 3)
    )

    if shift == 0:
        return sequence.copy()

    result = np.empty_like(sequence)

    if shift > 0:

        result[:shift] = sequence[0]

        result[shift:] = sequence[:-shift]

    else:

        shift = abs(shift)

        result[-shift:] = sequence[-1]

        result[:-shift] = sequence[shift:]

    return result


def augment_sequence(sequence, rng):

    augmented = temporal_shift(
        sequence,
        rng
    )

    # Small landmark noise
    noise = rng.normal(
        loc=0.0,
        scale=0.008,
        size=augmented.shape
    ).astype(np.float32)

    augmented = augmented + noise

    # Small scale variation
    scale = rng.uniform(
        0.97,
        1.03
    )

    augmented = augmented * scale

    return augmented.astype(np.float32)


def create_augmented_dataset(X, y):

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    X_augmented = []
    y_augmented = []

    for sequence, label in zip(X, y):

        # Original sample
        X_augmented.append(sequence)
        y_augmented.append(label)

        # Two augmented versions
        for _ in range(2):

            augmented = augment_sequence(
                sequence,
                rng
            )

            X_augmented.append(
                augmented
            )

            y_augmented.append(
                label
            )

    return (
        np.asarray(
            X_augmented,
            dtype=np.float32
        ),
        np.asarray(
            y_augmented,
            dtype=np.int32
        )
    )


# ============================================================
# MODEL
# ============================================================

def build_model(num_classes):

    inputs = tf.keras.Input(
        shape=(
            SEQUENCE_LENGTH,
            FEATURES_PER_FRAME
        ),
        name="landmark_sequence"
    )

    # Input regularization
    x = tf.keras.layers.GaussianNoise(
        0.01
    )(inputs)

    # Temporal feature extraction
    x = tf.keras.layers.Conv1D(
        filters=64,
        kernel_size=3,
        padding="same",
        activation="relu"
    )(x)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Dropout(
        0.20
    )(x)

    x = tf.keras.layers.Conv1D(
        filters=64,
        kernel_size=3,
        padding="same",
        activation="relu"
    )(x)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Dropout(
        0.20
    )(x)

    # Temporal motion learning
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            64,
            dropout=0.20
        )
    )(x)

    # Classification
    x = tf.keras.layers.Dense(
        64,
        activation="relu"
    )(x)

    x = tf.keras.layers.Dropout(
        0.35
    )(x)

    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
        name="word_prediction"
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="ISL_Word_Recognizer"
    )

    model.compile(

        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.0003
        ),

        loss="sparse_categorical_crossentropy",

        metrics=["accuracy"]
    )

    return model


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("ISL WORD MODEL TRAINING")
    print("=" * 60)

    # Reproducibility
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # LABELS
    # ========================================================

    class_dirs = sorted(
        [
            folder
            for folder in TRAIN_DIR.iterdir()
            if folder.is_dir()
        ]
    )

    labels = [
        folder.name
        for folder in class_dirs
    ]

    label_to_index = {
        label: index
        for index, label in enumerate(labels)
    }

    print("\nClasses:")

    for index, label in enumerate(labels):
        print(f"  {index}: {label}")

    # ========================================================
    # LOAD TRAINING DATA
    # ========================================================

    X_train_original, y_train_original = load_split(
        TRAIN_DIR,
        label_to_index
    )

    print(
        "\nOriginal training shape:",
        X_train_original.shape
    )

    # ========================================================
    # LOAD VALIDATION DATA
    # ========================================================

    X_val, y_val = load_split(
        VAL_DIR,
        label_to_index
    )

    print(
        "\nValidation shape:",
        X_val.shape
    )

    # ========================================================
    # LOAD TEST DATA
    # ========================================================

    X_test, y_test = load_split(
        TEST_DIR,
        label_to_index
    )

    print(
        "\nTest shape:",
        X_test.shape
    )

    # ========================================================
    # AUGMENT TRAINING DATA ONLY
    # ========================================================

    X_train, y_train = create_augmented_dataset(
        X_train_original,
        y_train_original
    )

    print(
        "\nTraining samples after augmentation:",
        len(X_train)
    )

    print(
        "Validation samples:",
        len(X_val)
    )

    print(
        "Test samples:",
        len(X_test)
    )

    # ========================================================
    # BUILD MODEL
    # ========================================================

    model = build_model(
        num_classes=len(labels)
    )

    print("\n")
    model.summary()

    # ========================================================
    # CALLBACKS
    # ========================================================

    callbacks = [

        tf.keras.callbacks.EarlyStopping(

            monitor="val_loss",

            patience=15,

            restore_best_weights=True,

            verbose=1
        ),

        tf.keras.callbacks.ReduceLROnPlateau(

            monitor="val_loss",

            factor=0.5,

            patience=5,

            min_lr=0.00001,

            verbose=1
        )
    ]

    # ========================================================
    # TRAIN
    # ========================================================

    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)

    history = model.fit(

        X_train,
        y_train,

        validation_data=(
            X_val,
            y_val
        ),

        epochs=EPOCHS,

        batch_size=BATCH_SIZE,

        callbacks=callbacks,

        shuffle=True,

        verbose=1
    )

    # ========================================================
    # VALIDATION EVALUATION
    # ========================================================

    val_loss, val_accuracy = model.evaluate(
        X_val,
        y_val,
        verbose=0
    )

    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)

    print(
        f"Validation loss     : {val_loss:.4f}"
    )

    print(
        f"Validation accuracy : {val_accuracy:.4f}"
    )

    # ========================================================
    # FINAL TEST EVALUATION
    # ========================================================

    print("\n" + "=" * 60)
    print("UNSEEN TEST RESULTS")
    print("=" * 60)

    test_loss, test_accuracy = model.evaluate(
        X_test,
        y_test,
        verbose=0
    )

    print(
        f"Test loss     : {test_loss:.4f}"
    )

    print(
        f"Test accuracy : {test_accuracy:.4f}"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "The test set was never used for training."
    )

    print(
        "This accuracy is our honest word-level result."
    )

    # ========================================================
    # SAVE MODEL
    # ========================================================

    model.save(
        MODEL_PATH
    )

    with open(
        LABELS_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            labels,
            file,
            indent=2,
            ensure_ascii=False
        )

    print("\n" + "=" * 60)
    print("MODEL SAVED")
    print("=" * 60)

    print(
        f"\nModel:"
    )

    print(
        MODEL_PATH
    )

    print(
        f"\nLabels:"
    )

    print(
        LABELS_PATH
    )


if __name__ == "__main__":
    main()
    