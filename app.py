import os
import base64
import numpy as np
import cv2
import joblib
from flask import Flask, render_template, request
from scipy.stats import skew, kurtosis
from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

CLASSES = ["Normal", "Bacterial pneumonia", "Viral pneumonia"]
ALLOWED = (".jpg", ".jpeg", ".png")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE_DIR, "pneumonia_classical_ml.joblib")

if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError("pneumonia_classical_ml.joblib not found next to app.py")

MODEL = joblib.load(MODEL_FILE)
CLASSIFIER = MODEL.named_steps["clf"].__class__.__name__ if hasattr(MODEL, "named_steps") else type(MODEL).__name__
print("Loaded model:", CLASSIFIER)


def extract_features(raw):
    data = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(data, (128, 128), interpolation=cv2.INTER_AREA)
    img = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(img)

    hog_feat = hog(img, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2), block_norm="L2-Hys")

    lbp_feat = []
    for points, radius in [(8, 1), (16, 2)]:
        lbp = local_binary_pattern(img, points, radius, method="uniform")
        hist, _ = np.histogram(lbp, bins=points + 2, range=(0, points + 2), density=True)
        lbp_feat.append(hist)

    quantized = (img // 4).astype(np.uint8)
    glcm = graycomatrix(quantized, distances=[1, 3], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                        levels=64, symmetric=True, normed=True)
    props = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]
    glcm_feat = np.concatenate([graycoprops(glcm, p).mean(axis=1) for p in props])

    pixels = img.ravel().astype(np.float32)
    stats = [pixels.mean(), pixels.std(), skew(pixels), kurtosis(pixels)]
    stats += list(np.percentile(pixels, [10, 25, 50, 75, 90]))
    hist, _ = np.histogram(pixels, bins=32, range=(0, 256), density=True)

    return np.concatenate([hog_feat, *lbp_feat, glcm_feat, stats, hist]).astype(np.float32)


def predict(raw):
    return MODEL.predict_proba(extract_features(raw).reshape(1, -1))[0]


@app.route("/", methods=["GET", "POST"])
def index():
    context = {"model_name": CLASSIFIER, "result": None, "error": None}

    if request.method == "POST":
        file = request.files.get("xray")
        if file is None or file.filename == "":
            context["error"] = "Please choose an X-ray image first."
        elif not file.filename.lower().endswith(ALLOWED):
            context["error"] = "Only JPG, JPEG and PNG images are supported."
        else:
            raw = file.read()
            try:
                probs = predict(raw)
            except Exception:
                context["error"] = "That file could not be read as an image."
            else:
                best = int(np.argmax(probs))
                mime = "image/png" if file.filename.lower().endswith(".png") else "image/jpeg"
                context["result"] = {
                    "label": CLASSES[best],
                    "confidence": float(probs[best]) * 100,
                    "rows": [(name, float(p) * 100) for name, p in zip(CLASSES, probs)],
                    "image": "data:" + mime + ";base64," + base64.b64encode(raw).decode(),
                    "is_normal": best == 0,
                }

    return render_template("index.html", **context)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
