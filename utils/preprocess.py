import cv2
import numpy as np


PREPROCESSING = {
    "cnn": {"size": 128, "normalization": "zero_one"},
    "mobilenet": {"size": 224, "normalization": "mobilenet"},
}


def preprocess_image(image_path, model_type: str) -> np.ndarray:
    """Reproduce the notebook's decode, resize, and model-specific scaling."""
    if model_type not in PREPROCESSING:
        raise ValueError(f"Unsupported model type: {model_type}")

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ValueError("The image could not be read by OpenCV.")

    # OpenCV reads BGR; TensorFlow's decoder produces RGB image tensors.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    size = PREPROCESSING[model_type]["size"]
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    image = image.astype(np.float32)

    if PREPROCESSING[model_type]["normalization"] == "mobilenet":
        image = (image / 127.5) - 1.0
    else:
        image = image / 255.0

    return np.expand_dims(image, axis=0)
