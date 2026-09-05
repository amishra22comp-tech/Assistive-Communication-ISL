import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from src.config import PROCESSED_DATASET_DIR, MODEL_PATH, LABELS_PATH, TARGET_SENTENCES
from src.train_model import load_data
from src.config import SEQUENCE_LENGTH, FEATURES_PER_FRAME

def main():
    if not MODEL_PATH.exists() or not LABELS_PATH.exists():
        raise RuntimeError("Train the model first.")

    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABELS_PATH, encoding="utf-8") as f:
        labels = json.load(f)

    X, y_text = load_data()
    label_to_index = {label: i for i, label in enumerate(labels)}
    y_true = np.array([label_to_index[y] for y in y_text])

    probabilities = model.predict(X, verbose=0)
    y_pred = probabilities.argmax(axis=1)

    print(classification_report(
        y_true, y_pred,
        target_names=labels,
        zero_division=0
    ))
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    main()
