import json
from collections import deque
import numpy as np
import tensorflow as tf

from src.config import MODEL_PATH, LABELS_PATH, SEQUENCE_LENGTH, MIN_CONFIDENCE
from src.preprocess import prepare_sequence

class SequencePredictor:
    def __init__(self):
        self.model = None
        self.labels = []
        self.buffer = deque(maxlen=SEQUENCE_LENGTH)

    def load(self):
        if self.model is None:
            if not MODEL_PATH.exists() or not LABELS_PATH.exists():
                raise RuntimeError("Model not trained yet.")
            self.model = tf.keras.models.load_model(MODEL_PATH)
            self.labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))

    def reset(self):
        self.buffer.clear()

    def add_features(self, features):
        self.load()
        self.buffer.append(np.asarray(features, dtype=np.float32))

        if len(self.buffer) < SEQUENCE_LENGTH:
            return {
                "success": False,
                "message": f"Collecting frames ({len(self.buffer)}/{SEQUENCE_LENGTH})"
            }

        sequence = prepare_sequence(np.asarray(self.buffer))
        probs = self.model.predict(sequence[None, ...], verbose=0)[0]
        idx = int(np.argmax(probs))
        confidence = float(probs[idx])

        if confidence < MIN_CONFIDENCE:
            return {
                "success": False,
                "message": "No supported sign detected"
            }

        return {
            "success": True,
            "sentence": self.labels[idx],
            "confidence": round(confidence * 100, 2),
        }

predictor = SequencePredictor()
