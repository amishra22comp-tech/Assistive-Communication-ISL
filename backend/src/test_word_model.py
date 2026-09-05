import os
import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

TEST_DIR = os.path.join(
    BASE_DIR, "dataset", "word_processed", "test"
)

MODEL_PATH = os.path.join(
    BASE_DIR, "models", "isl_word_model.keras"
)

LABEL_PATH = os.path.join(
    BASE_DIR, "models", "word_labels.json"
)


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 60)
print("LOADING WORD-LEVEL ISL MODEL")
print("=" * 60)

model = tf.keras.models.load_model(MODEL_PATH)

print(f"\nModel loaded successfully:")
print(MODEL_PATH)


# ============================================================
# LOAD LABELS
# ============================================================

with open(LABEL_PATH, "r") as f:
    labels = json.load(f)

print("\nLabels:")

for index, label in enumerate(labels):
    print(f"{index}: {label}")


# ============================================================
# LOAD TEST DATA
# ============================================================

X_test = []
y_test = []


print("\n" + "=" * 60)
print("LOADING UNSEEN TEST DATA")
print("=" * 60)


for label_index, label in enumerate(labels):

    class_dir = os.path.join(TEST_DIR, label)

    if not os.path.exists(class_dir):
        print(f"WARNING: Missing folder: {label}")
        continue

    files = [
        file for file in os.listdir(class_dir)
        if file.endswith(".npy")
    ]

    print(f"{label:15s}: {len(files)} samples")

    for file in files:

        file_path = os.path.join(class_dir, file)

        sequence = np.load(file_path)

        X_test.append(sequence)
        y_test.append(label_index)


X_test = np.array(X_test, dtype=np.float32)
y_test = np.array(y_test)


print("\nTest data shape:", X_test.shape)
print("Test labels shape:", y_test.shape)
print("Total test samples:", len(X_test))


# ============================================================
# PREDICTIONS
# ============================================================

print("\n" + "=" * 60)
print("RUNNING PREDICTIONS")
print("=" * 60)

predictions = model.predict(X_test, verbose=1)

y_pred = np.argmax(predictions, axis=1)


# ============================================================
# OVERALL ACCURACY
# ============================================================

accuracy = np.mean(y_pred == y_test)

print("\n" + "=" * 60)
print("OVERALL TEST RESULT")
print("=" * 60)

print(f"\nTest Accuracy: {accuracy * 100:.2f}%")


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

report = classification_report(
    y_test,
    y_pred,
    labels=list(range(len(labels))),
    target_names=labels,
    zero_division=0
)

print(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=list(range(len(labels)))
)

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print("\nRows = Actual")
print("Columns = Predicted\n")

print(" " * 15 + " ".join(f"{i:4}" for i in range(len(labels))))

for i, row in enumerate(cm):

    print(
        f"{labels[i]:15s} "
        + " ".join(f"{value:4}" for value in row)
    )


# ============================================================
# WRONG PREDICTIONS
# ============================================================

print("\n" + "=" * 60)
print("WRONG PREDICTIONS")
print("=" * 60)

wrong_count = 0

for actual, predicted in zip(y_test, y_pred):

    if actual != predicted:

        print(
            f"Actual: {labels[actual]:15s}"
            f" -> Predicted: {labels[predicted]}"
        )

        wrong_count += 1


print("\nTotal incorrect predictions:", wrong_count)
print("Total correct predictions:", len(y_test) - wrong_count)


print("\n" + "=" * 60)
print("TESTING COMPLETE")
print("=" * 60)