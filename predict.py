from functools import lru_cache

import numpy as np
import tensorflow as tf

from config import CLASS_LABELS, MODEL_PATH, MODEL_TYPE
from utils.preprocess import preprocess_image


@lru_cache(maxsize=1)
def get_model():
    """Load the Keras model once per Flask process."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_PATH}. Copy your saved .keras model there."
        )
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


def predict_image(image_path: str) -> dict:
    """Return the predicted label, confidence, and all class probabilities."""
    batch = preprocess_image(image_path, MODEL_TYPE)
    probabilities = get_model().predict(batch, verbose=0)[0]
    probabilities = np.asarray(probabilities, dtype=float)

    if probabilities.shape[0] != len(CLASS_LABELS):
        raise ValueError("The model output does not match the configured four class labels.")

    index = int(np.argmax(probabilities))
    all_probabilities = [
        {"label": label, "percent": round(float(probability) * 100, 2)}
        for label, probability in zip(CLASS_LABELS, probabilities)
    ]
    return {
        "label": CLASS_LABELS[index],
        "confidence": round(float(probabilities[index]) * 100, 2),
        "probabilities": all_probabilities,
    }
