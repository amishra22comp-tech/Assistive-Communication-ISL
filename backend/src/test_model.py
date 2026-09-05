import json
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.config import (
    PROCESSED_DATASET_DIR,
    MODEL_PATH,
    LABELS_PATH,
    TARGET_SENTENCES,
    MIN_CONFIDENCE,
    SEQUENCE_LENGTH,
    FEATURES_PER_FRAME,
)


# Must exactly match train_model.py
RANDOM_STATE = 42
VALIDATION_SIZE = 0.25


def load_data():
    X = []
    y = []
    file_paths = []

    print("\nLoading processed dataset...\n")

    for sentence in TARGET_SENTENCES:
        class_dir = PROCESSED_DATASET_DIR / sentence
        files = sorted(class_dir.glob("*.npy"))

        if not files:
            print(f"WARNING: No samples found for: {sentence}")
            continue

        for file in files:
            sequence = np.load(file)

            expected_shape = (
                SEQUENCE_LENGTH,
                FEATURES_PER_FRAME,
            )

            if sequence.shape != expected_shape:
                print(
                    f"Skipping bad shape: {file} | "
                    f"Found: {sequence.shape}"
                )
                continue

            X.append(sequence.astype(np.float32))
            y.append(sentence)
            file_paths.append(file)

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y),
        np.asarray(file_paths, dtype=object),
    )


def main():

    print("\n" + "=" * 60)
    print("UNSEEN VALIDATION TEST")
    print("=" * 60)

    # Load trained model
    model = tf.keras.models.load_model(MODEL_PATH)

    # Load labels saved during training
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        labels = json.load(f)

    # Load all original processed videos
    X, y_labels, file_paths = load_data()

    # Recreate the exact label encoding
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y_labels)

    # Recreate the exact same split used in training
    (
        _,
        X_val,
        _,
        y_val,
        _,
        val_files,
    ) = train_test_split(
        X,
        y_encoded,
        file_paths,
        test_size=VALIDATION_SIZE,
        stratify=y_encoded,
        random_state=RANDOM_STATE,
    )

    print(f"\nLoaded model: {MODEL_PATH}")
    print(f"Total dataset videos: {len(X)}")
    print(f"Validation videos only: {len(X_val)}")
    print(f"Training videos excluded: {len(X) - len(X_val)}")
    print("=" * 60)

    # Track results per sentence
    sentence_total = {
        sentence: 0
        for sentence in labels
    }

    sentence_correct = {
        sentence: 0
        for sentence in labels
    }

    total_correct = 0

    for sequence, true_index, file in zip(
        X_val,
        y_val,
        val_files,
    ):

        true_label = labels[int(true_index)]

        probabilities = model.predict(
            sequence[np.newaxis, ...],
            verbose=0,
        )[0]

        predicted_index = int(
            np.argmax(probabilities)
        )

        predicted_label = labels[
            predicted_index
        ]

        confidence = float(
            probabilities[predicted_index]
        )

        status = (
            "CORRECT"
            if predicted_index == int(true_index)
            else "WRONG"
        )

        confidence_status = (
            "ACCEPTED"
            if confidence >= MIN_CONFIDENCE
            else "LOW CONFIDENCE"
        )

        print(
            f"\nFile: {file.name}"
        )

        print(
            f"Expected: {true_label}"
        )

        print(
            f"Predicted: {predicted_label}"
        )

        print(
            f"Confidence: {confidence:.2%}"
        )

        print(
            f"Result: {status} | "
            f"{confidence_status}"
        )

        sentence_total[true_label] += 1

        if predicted_index == int(true_index):
            total_correct += 1
            sentence_correct[true_label] += 1

    # Overall result
    accuracy = (
        total_correct / len(X_val)
    ) * 100

    print("\n" + "=" * 60)
    print("PER-SENTENCE VALIDATION RESULTS")
    print("=" * 60)

    for sentence in labels:

        total = sentence_total[sentence]
        correct = sentence_correct[sentence]

        if total == 0:
            continue

        sentence_accuracy = (
            correct / total
        ) * 100

        print(
            f"{sentence}: "
            f"{correct}/{total} | "
            f"{sentence_accuracy:.2f}%"
        )

    print("\n" + "=" * 60)
    print("FINAL UNSEEN VALIDATION ACCURACY")
    print("=" * 60)

    print(
        f"Correct predictions: "
        f"{total_correct}/{len(X_val)}"
    )

    print(
        f"Accuracy: {accuracy:.2f}%"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()