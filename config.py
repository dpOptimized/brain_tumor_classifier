from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "brain_tumor_model.keras"
UPLOAD_FOLDER = BASE_DIR / "uploads"

# This notebook saved two independent models.  Use "cnn" for model.keras and
# "mobilenet" for mobilenet_v2.keras after copying the selected file above.
MODEL_TYPE = "cnn"

CLASS_LABELS = ("glioma", "meningioma", "pituitary", "notumor")
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10 MB
SECRET_KEY = "replace-this-with-a-random-secret-key"
