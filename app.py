from pathlib import Path
from uuid import uuid4

from flask import Flask, render_template, request, send_from_directory, url_for
from werkzeug.exceptions import RequestEntityTooLarge

from config import MAX_CONTENT_LENGTH, SECRET_KEY, UPLOAD_FOLDER
from predict import predict_image
from utils.validator import UploadValidationError, validate_upload


def create_app():
    app = Flask(__name__)
    app.config.update(
        SECRET_KEY=SECRET_KEY,
        UPLOAD_FOLDER=str(UPLOAD_FOLDER),
        MAX_CONTENT_LENGTH=MAX_CONTENT_LENGTH,
    )
    UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

    @app.get("/")
    def home():
        return render_template("home.html")

    @app.get("/uploads/<path:filename>")
    def uploaded_file(filename):
        return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

    @app.post("/predict")
    def predict():
        image_file = request.files.get("image")
        try:
            filename = validate_upload(image_file)
            stored_name = f"{uuid4().hex}_{filename}"
            image_path = Path(app.config["UPLOAD_FOLDER"]) / stored_name
            image_file.save(image_path)
            result = predict_image(str(image_path))
            image_url = url_for("uploaded_file", filename=stored_name)
            return render_template("prediction.html", result=result, image_url=image_url)
        except (UploadValidationError, ValueError, FileNotFoundError) as error:
            return render_template("error.html", message=str(error)), 400
        except Exception:
            app.logger.exception("Prediction failed")
            return render_template(
                "error.html", message="The image could not be processed. Please try another MRI image."
            ), 500

    @app.errorhandler(RequestEntityTooLarge)
    def file_too_large(_error):
        return render_template("error.html", message="The image must be 10 MB or smaller."), 413

    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=True)
