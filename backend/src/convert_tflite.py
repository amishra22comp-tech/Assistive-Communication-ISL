import tensorflow as tf
from src.config import MODEL_PATH, MODEL_DIR

def main():
    model = tf.keras.models.load_model(MODEL_PATH)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    output = MODEL_DIR / "isl_sentence_model.tflite"
    output.write_bytes(tflite_model)
    print("Saved:", output)

if __name__ == "__main__":
    main()
