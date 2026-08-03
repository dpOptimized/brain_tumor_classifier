# Brain Tumor Classifier

## Overview

A Flask web application that performs inference on brain MRI images using a trained Keras model. It does not load datasets or train, validate, test, or plot models.

The implementation follows `brain_tumor_training.ipynb`:

- Classes: `glioma`, `meningioma`, `pituitary`, `notumor`
- Custom CNN: 128 × 128 RGB, pixel values divided by 255
- MobileNetV2: 224 × 224 RGB, `(pixel / 127.5) - 1`

## Features

- JPG, JPEG, and PNG uploads only
- Rejects unsupported, empty, corrupted, unreadable, and unprocessable files
- Loads the Keras model only once per Flask process
- Shows prediction, confidence, per-class probabilities, and the uploaded MRI
- Simple responsive Bootstrap 5 interface

## Folder Structure

```text
brain-tumor-classifier/
├── app.py
├── config.py
├── predict.py
├── requirements.txt
├── README.md
├── model/
│   └── brain_tumor_model.keras
├── notebooks/
│   └── brain_tumor_training.ipynb
├── templates/
│   ├── layout.html
│   ├── home.html
│   ├── prediction.html
│   └── error.html
├── static/
│   └── style.css
├── uploads/
└── utils/
    ├── preprocess.py
    └── validator.py
```

## Installation

1. Create and activate a virtual environment.

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. Install dependencies.

   ```powershell
   pip install -r requirements.txt
   ```

3. Copy the trained model from the notebook environment to `model/brain_tumor_model.keras`.

   For the notebook's custom CNN, copy `model.keras` and leave `MODEL_TYPE = "cnn"` in `config.py`.

   For the notebook's MobileNetV2 model, copy `mobilenet_v2.keras` under the same destination name and set `MODEL_TYPE = "mobilenet"` in `config.py`.

## Usage

```powershell
python app.py
```

Open `http://127.0.0.1:5000`, upload a JPG, JPEG, or PNG MRI image, and select **Predict**.

## Technologies Used

- Python 3
- Flask
- TensorFlow / Keras
- OpenCV
- NumPy
- Bootstrap 5


## Future Improvements

- Add authenticated access and audit logging for clinical workflows.
- Add automatic cleanup for stored upload images.
- Add Docker deployment configuration.
- Support the notebook's two-model averaging ensemble by deploying both saved models.
- Add unit and integration tests.

> This application is for educational and research use. It is not a medical device and must not be used as a substitute for clinical diagnosis.