"""
AlbaID — Flask Backend
Sindi Gugalli & Inviona Hoxha | Introduction to Computer Graphics
"""

import os
import uuid
import base64
import time
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
from utils import extract_face, compute_orb_score, verify_faces, ORB_THRESHOLD

app = Flask(__name__, template_folder="../templates", static_folder="../static")
CORS(app)

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "../static/uploads")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/verify", methods=["POST"])
def verify():
    """
    Accepts multipart form data with:
      - selfie: image file
      - id_card: image file
    Returns JSON with full verification result.
    Uses ORB similarity scoring (threshold derived in analysis.ipynb).
    """
    if "selfie" not in request.files or "id_card" not in request.files:
        return jsonify({"error": "Both 'selfie' and 'id_card' files are required."}), 400

    selfie_file = request.files["selfie"]
    id_card_file = request.files["id_card"]

    if not (allowed_file(selfie_file.filename) and allowed_file(id_card_file.filename)):
        return jsonify({"error": "Only PNG, JPG, JPEG, WEBP files are allowed."}), 400

    uid = str(uuid.uuid4())[:8]
    selfie_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{uid}_selfie.jpg")
    id_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{uid}_id.jpg")

    selfie_file.save(selfie_path)
    id_card_file.save(id_path)

    try:
        import cv2
        t0 = time.time()

        selfie_bgr = cv2.imread(selfie_path)
        id_bgr = cv2.imread(id_path)

        if selfie_bgr is None or id_bgr is None:
            return jsonify({"error": "Could not read one or both image files."}), 500

        selfie_face, selfie_bbox = extract_face(selfie_bgr)
        id_face, id_bbox = extract_face(id_bgr)

        selfie_detected = selfie_face is not None
        id_detected = id_face is not None

        if not selfie_detected or not id_detected:
            result = {
                "verdict": "ERROR",
                "confidencePercent": 0.0,
                "orbScore": 0.0,
                "threshold": ORB_THRESHOLD,
                "biometrics": {
                    "faceDetectedSelfie": selfie_detected,
                    "faceDetectedID": id_detected,
                },
                "document": {
                    "idCardDetected": id_detected,
                    "perspectiveCorrectionApplied": False,
                },
                "reportNarrative": (
                    f"Face detection failed on "
                    f"{'selfie' if not selfie_detected else 'ID card'} image(s). "
                    f"Please ensure faces are clearly visible and well-lit."
                ),
                "processingTimeMs": 0.0,
                "selfie_url": f"/static/uploads/{uid}_selfie.jpg",
                "id_url": f"/static/uploads/{uid}_id.jpg",
            }
            return jsonify(result)

        score = compute_orb_score(selfie_face, id_face)
        verification = verify_faces(selfie_face, id_face)

        confidence = 0.0
        if verification["verdict"] == "MATCH":
            confidence = min((score / ORB_THRESHOLD) * 100, 99.0)
        elif verification["verdict"] == "MISMATCH":
            confidence = max(0.0, (1.0 - (score / 100.0)) * 30)

        processing_ms = (time.time() - t0) * 1000

        result = {
            "verdict": verification["verdict"],
            "confidencePercent": round(confidence, 1),
            "orbScore": round(score, 1),
            "threshold": ORB_THRESHOLD,
            "biometrics": {
                "faceDetectedSelfie": selfie_detected,
                "faceDetectedID": id_detected,
            },
            "document": {
                "idCardDetected": id_detected,
                "perspectiveCorrectionApplied": False,
            },
            "reportNarrative": (
                f"Both faces were successfully detected and cropped to 220x220px. "
                f"ORB descriptor matching yielded a similarity score of {score:.1f}/100 "
                f"(threshold: {ORB_THRESHOLD}). "
                f"{'Identity MATCH confirmed.' if verification['verdict'] == 'MATCH' else 'Identity MISMATCH — signatures do not correspond.'}"
            ),
            "processingTimeMs": round(processing_ms, 1),
            "selfie_url": f"/static/uploads/{uid}_selfie.jpg",
            "id_url": f"/static/uploads/{uid}_id.jpg",
        }
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/verify_base64", methods=["POST"])
def verify_base64():
    """
    Accepts JSON body with:
      - selfie_b64: base64-encoded image string
      - id_b64: base64-encoded image string
    Returns JSON with full verification result.
    """
    data = request.get_json()
    if not data or "selfie_b64" not in data or "id_b64" not in data:
        return jsonify({"error": "Both 'selfie_b64' and 'id_b64' are required."}), 400

    uid = str(uuid.uuid4())[:8]
    selfie_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{uid}_selfie.jpg")
    id_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{uid}_id.jpg")

    def save_b64(b64_str, path):
        if "," in b64_str:
            b64_str = b64_str.split(",")[1]
        with open(path, "wb") as f:
            f.write(base64.b64decode(b64_str))

    save_b64(data["selfie_b64"], selfie_path)
    save_b64(data["id_b64"], id_path)

    try:
        import cv2
        selfie_bgr = cv2.imread(selfie_path)
        id_bgr = cv2.imread(id_path)

        selfie_face, _ = extract_face(selfie_bgr)
        id_face, _ = extract_face(id_bgr)

        score = compute_orb_score(selfie_face, id_face) if selfie_face and id_face else 0.0
        verification = verify_faces(selfie_face, id_face)

        return jsonify({
            "verdict": verification["verdict"],
            "orbScore": round(score, 1),
            "threshold": ORB_THRESHOLD,
            "confidencePercent": round(
                min((score / ORB_THRESHOLD) * 100, 99.0) if verification["verdict"] == "MATCH"
                else max(0.0, (1.0 - score / 100.0) * 30),
                1
            ),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    return jsonify({"status": "ok", "version": "2.0.0", "method": "ORB"})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)