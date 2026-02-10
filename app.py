# Created: 2026-02-10
"""
Web version of Handwritten Digit Recognition.
Flask backend that serves the drawing UI and handles prediction requests.
"""

import os
import sys
import base64
import io
import pickle

import numpy as np
from PIL import Image, ImageFilter
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Model path: check current dir first, then parent dir (for local dev)
_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_DIR, "digit_model.pkl")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(_DIR, "..", "digit_model.pkl")
IMG_SIZE = 28

# Load model at startup
model = None


def load_model():
    """Load the trained model from disk."""
    global model
    if os.path.exists(MODEL_PATH):
        print(f"Loading model from {MODEL_PATH}...")
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        print("Model loaded successfully.")
    else:
        print(f"ERROR: Model file not found.")
        print("Please place digit_model.pkl in the same directory as app.py.")
        sys.exit(1)


def preprocess_image(img):
    """Convert a PIL image to a 28x28 MNIST-compatible format.

    Replicates the desktop version's _preprocess_image() logic exactly.
    """
    # Convert to grayscale if needed
    if img.mode != "L":
        img = img.convert("L")

    # Find the bounding box of the drawn content
    bbox = img.getbbox()
    if bbox is None:
        return None

    # Crop to content with some padding
    left, top, right, bottom = bbox
    width = right - left
    height = bottom - top

    # Add padding (20% of the larger dimension)
    pad = int(max(width, height) * 0.2)
    img_width, img_height = img.size
    left = max(0, left - pad)
    top = max(0, top - pad)
    right = min(img_width, right + pad)
    bottom = min(img_height, bottom + pad)

    img = img.crop((left, top, right, bottom))

    # Make it square by adding padding to the shorter side
    width = right - left
    height = bottom - top
    max_dim = max(width, height)
    new_img = Image.new("L", (max_dim, max_dim), 0)
    offset_x = (max_dim - width) // 2
    offset_y = (max_dim - height) // 2
    new_img.paste(img, (offset_x, offset_y))

    # Resize to 28x28 (MNIST format)
    img_resized = new_img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)

    # Apply a slight Gaussian blur to smooth edges (like MNIST data)
    img_resized = img_resized.filter(ImageFilter.GaussianBlur(radius=0.5))

    # Convert to numpy array and flatten
    pixel_data = np.array(img_resized, dtype=np.float64).reshape(1, -1)

    return pixel_data


@app.route("/")
def index():
    """Serve the main drawing page."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Receive a Base64 image, preprocess it, and return prediction."""
    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"error": "No image data provided"}), 400

    # Decode Base64 image
    image_data = data["image"]
    # Strip the data URL prefix if present
    if "," in image_data:
        image_data = image_data.split(",", 1)[1]

    try:
        image_bytes = base64.b64decode(image_data)
        img = Image.open(io.BytesIO(image_bytes)).convert("L")
    except Exception:
        return jsonify({"error": "Invalid image data"}), 400

    # Preprocess
    pixel_data = preprocess_image(img)
    if pixel_data is None:
        return jsonify({"error": "No drawing detected"}), 400

    # Predict
    prediction = int(model.predict(pixel_data)[0])
    probabilities = model.predict_proba(pixel_data)[0].tolist()
    confidence = probabilities[prediction] * 100

    return jsonify({
        "prediction": prediction,
        "confidence": round(confidence, 1),
        "probabilities": [round(p * 100, 1) for p in probabilities],
    })


load_model()

if __name__ == "__main__":
    print("Starting web server at http://localhost:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
