import os
import sys
import subprocess
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS, cross_origin

from cnnClassifier.utils.common import decodeImage
from cnnClassifier.pipeline.predict import PredictionPipeline

os.environ.setdefault("LANG", "en_US.UTF-8")
os.environ.setdefault("LC_ALL", "en_US.UTF-8")

app = Flask(__name__)
CORS(app)


class ClientApp:
    def __init__(self) -> None:
        self.filename = "inputImage.jpg"
        self.classifier = PredictionPipeline(self.filename)


clApp: "ClientApp"


@app.route("/", methods=["GET"])
@cross_origin()
def home():
    try:
        return render_template("index.html")
    except Exception:
        return "Chicken Disease Classifier API is running."


@app.route("/train", methods=["GET", "POST"])
@cross_origin()
def trainRoute():
    try:
        subprocess.run([sys.executable, "main.py"], check=True)
        return "Training done successfully!"
    except subprocess.CalledProcessError as e:
        return f"Training failed: {e}", 500


@app.route("/predict", methods=["POST"])
@cross_origin()
def predictRoute():
    data = request.get_json(silent=True) or {}
    image_b64 = data.get("image")
    if not image_b64:
        return jsonify({"error": "Missing 'image' field"}), 400

    decodeImage(image_b64, clApp.filename)
    result = clApp.classifier.predict()
    return jsonify(result)


if __name__ == "__main__":
    clApp = ClientApp()
    app.run(host="0.0.0.0", port=8080, debug=True)

