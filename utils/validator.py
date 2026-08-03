from pathlib import Path

import cv2
import numpy as np
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

from config import ALLOWED_EXTENSIONS


class UploadValidationError(ValueError):
    """Raised when an uploaded file is not a usable MRI image."""


def validate_upload(file: FileStorage) -> str:
    """Validate extension and image bytes, returning a safe file name."""
    if file is None or not file.filename:
        raise UploadValidationError("Please choose an MRI image to upload.")

    filename = secure_filename(file.filename)
    extension = Path(filename).suffix.lower().lstrip(".")
    if not filename or extension not in ALLOWED_EXTENSIONS:
        raise UploadValidationError("Only JPG, JPEG, and PNG image files are allowed.")

    data = file.read()
    file.seek(0)
    if not data:
        raise UploadValidationError("The uploaded image is empty.")

    image_array = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise UploadValidationError(
            "The uploaded file is corrupted, unreadable, or cannot be processed as an image."
        )

    return filename
