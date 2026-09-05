import json
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.config import (
    PROCESSED_DATASET_DIR,
    MODEL_DIR,
    MODEL_PATH,
    LABELS_PATH,
    TARGET_SENTENCES,
    SEQUENCE_LENGTH,
    FEATURES_PER_FRAME,
)


# ==================================================
# TRAINING SETTINGS
# ==================================================

EPOCHS = 150
BATCH_SIZE = 8
RANDOM_STATE = 42

# Keep validation videos completely untouched
VALIDATION_SIZE = 0.25


# ==================================================
# LOAD DATA
# ==================================================

def load_data():

    X = []
    y = []

    print("\nLoading processed dataset...\n")

    for sentence in TARGET_SENTENCES:

        class_dir = (
            PROCESSED_DATASET_DIR /
            sentence
        )

        if not class_dir.exists():

            print(
                f"WARNING: Missing folder: "
                f"{class_dir}"
            )

            continue

        files = sorted(
            class_dir.glob("*.npy")
        )

        print(
            f"{sentence}: "
            f"{len(files)} samples"
        )

        for file in files:

            arr = np.load(file)

            expected_shape = (
                SEQUENCE_LENGTH,
                FEATURES_PER_FRAME,
            )

            if arr.shape != expected_shape:

                print(
                    f"Skipping bad shape: "
                    f"{file}"
                )

                print(
                    f"Expected: {expected_shape}"
                )

                print(
                    f"Found: {arr.shape}"
                )

                continue

            X.append(
                arr.astype(np.float32)
            )

            y.append(sentence)

    if not X:

        raise RuntimeError(
            "No processed samples found."
        )

    return (
        np.asarray(
            X,
            dtype=np.float32,
        ),
        np.asarray(y),
    )


# ==================================================
# AUGMENTATION
# ==================================================

def temporal_shift(
    sequence,
    rng,
):

    shift = int(
        rng.integers(-2, 3)
    )

    if shift == 0:

        return sequence.copy()

    result = np.empty_like(
        sequence
    )

    if shift > 0:

        result[:shift] = sequence[0]

        result[shift:] = (
            sequence[:-shift]
        )

    else:

        shift = abs(shift)

        result[-shift:] = sequence[-1]

        result[:-shift] = (
            sequence[shift:]
        )

    return result


def augment_sequence(
    sequence,
    rng,
):

    augmented = temporal_shift(
        sequence,
        rng,
    )

    # Small landmark noise
    noise = rng.normal(
        loc=0.0,
        scale=0.01,
        size=augmented.shape,
    ).astype(np.float32)

    augmented = (
        augmented + noise
    )

    # Small scale variation
    scale = rng.uniform(
        0.95,
        1.05,
    )

    augmented = (
        augmented * scale
    )

    return augmented.astype(
        np.float32
    )


def create_augmented_dataset(
    X,
    y,
):

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    X_augmented = []
    y_augmented = []

    for sequence, label in zip(X, y):

        # Original sample
        X_augmented.append(
            sequence
        )

        y_augmented.append(
            label
        )

        # Four augmented versions
        for _ in range(4):

            augmented = (
                augment_sequence(
                    sequence,
                    rng,
                )
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
            dtype=np.float32,
        ),
        np.asarray(
            y_augmented,
            dtype=np.int32,
        ),
    )


# ==================================================
# MODEL
# ==================================================

def build_model(
    num_classes,
):

    inputs = tf.keras.Input(
        shape=(
            SEQUENCE_LENGTH,
            FEATURES_PER_FRAME,
        ),
        name="landmark_sequence",
    )

    # Input noise improves generalization
    x = tf.keras.layers.GaussianNoise(
        0.01
    )(inputs)

    # Temporal feature extraction
    x = tf.keras.layers.Conv1D(
        filters=64,
        kernel_size=3,
        padding="same",
        activation="relu",
    )(x)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Dropout(
        0.20
    )(x)

    x = tf.keras.layers.Conv1D(
        filters=64,
        kernel_size=3,
        padding="same",
        activation="relu",
    )(x)

    x = tf.keras.layers.BatchNormalization()(x)

    # Learn temporal motion patterns
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            64,
            dropout=0.20,
        )
    )(x)

    # Classification layers
    x = tf.keras.layers.Dense(
        64,
        activation="relu",
    )(x)

    x = tf.keras.layers.Dropout(
        0.40
    )(x)

    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
        name="sentence_prediction",
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="ISL_Sentence_Recognizer",
    )

    model.compile(

        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.0003
        ),

        loss=(
            "sparse_categorical_crossentropy"
        ),

        metrics=[
            "accuracy"
        ],
    )

    return model


# ==================================================
# MAIN
# ==================================================

def main():

    # Reproducibility
    np.random.seed(
        RANDOM_STATE
    )

    tf.random.set_seed(
        RANDOM_STATE
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ----------------------------------------------
    # Load data
    # ----------------------------------------------

    X_original, labels_original = (
        load_data()
    )

    print(
        "\nOriginal dataset shape:",
        X_original.shape,
    )

    # ----------------------------------------------
    # Encode labels
    # ----------------------------------------------

    encoder = LabelEncoder()

    y_encoded = encoder.fit_transform(
        labels_original
    )

    print(
        "\nClasses:",
        encoder.classes_.tolist(),
    )

    # ----------------------------------------------
    # Split REAL videos first
    # ----------------------------------------------

    (
        X_train_original,
        X_val,
        y_train,
        y_val,
    ) = train_test_split(

        X_original,
        y_encoded,

        test_size=VALIDATION_SIZE,

        stratify=y_encoded,

        random_state=RANDOM_STATE,
    )

    print(
        "\nTraining real videos:",
        len(X_train_original),
    )

    print(
        "Validation videos:",
        len(X_val),
    )

    # ----------------------------------------------
    # Augment ONLY training videos
    # ----------------------------------------------

    X_train, y_train_augmented = (
        create_augmented_dataset(

            X_train_original,

            y_train,
        )
    )

    print(
        "Training samples "
        "after augmentation:",
        len(X_train),
    )

    # ----------------------------------------------
    # Build model
    # ----------------------------------------------

    model = build_model(
        len(
            encoder.classes_
        )
    )

    print("\n")

    model.summary()

    # ----------------------------------------------
    # Callbacks
    # ----------------------------------------------

    callbacks = [

        tf.keras.callbacks.EarlyStopping(

            monitor="val_loss",

            patience=25,

            restore_best_weights=True,

            verbose=1,
        ),

        tf.keras.callbacks.ReduceLROnPlateau(

            monitor="val_loss",

            factor=0.5,

            patience=8,

            min_lr=0.00001,

            verbose=1,
        ),

    ]

    # ----------------------------------------------
    # Train
    # ----------------------------------------------

    history = model.fit(

        X_train,

        y_train_augmented,

        validation_data=(

            X_val,

            y_val,
        ),

        epochs=EPOCHS,

        batch_size=BATCH_SIZE,

        callbacks=callbacks,

        verbose=1,
    )

    # ----------------------------------------------
    # Final evaluation
    # ----------------------------------------------

    loss, accuracy = model.evaluate(

        X_val,

        y_val,

        verbose=0,
    )

    print("\n" + "=" * 50)

    print(
        f"FINAL VALIDATION LOSS: "
        f"{loss:.4f}"
    )

    print(
        f"FINAL VALIDATION ACCURACY: "
        f"{accuracy:.4f}"
    )

    print("=" * 50)

    # ----------------------------------------------
    # Save model
    # ----------------------------------------------

    model.save(
        MODEL_PATH
    )

    with open(
        LABELS_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(

            encoder.classes_.tolist(),

            file,

            indent=2,

            ensure_ascii=False,
        )

    print(
        "\nModel saved:"
    )

    print(
        MODEL_PATH
    )

    print(
        "\nLabels saved:"
    )

    print(
        LABELS_PATH
    )


if __name__ == "__main__":

    main()