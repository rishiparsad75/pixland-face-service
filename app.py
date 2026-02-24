"""
PixLand Face Recognition Microservice
======================================
Uses DeepFace (ArcFace model) for production-grade face recognition.
ArcFace produces 512-dimensional L2-normalized embeddings.

Endpoints:
  POST /extract   — Extract face embedding from uploaded image
  POST /compare   — Compare two embeddings or two images
  GET  /health    — Health check
"""

import os
import base64
import io
import logging
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image as PILImage

# Lazy import DeepFace to speed up startup
deepface = None

app = Flask(__name__)
CORS(app)  # Allow Node.js backend to call this service
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = "ArcFace"
DETECTOR = "opencv"        # Faster/Lighter for B1 plan
DISTANCE_METRIC = "cosine" # ArcFace + cosine = best combo


def load_deepface():
    """Lazy-load DeepFace on first use."""
    global deepface
    if deepface is None:
        logger.info("[Init] Loading DeepFace...")
        from deepface import DeepFace as df
        deepface = df
        # Warm-up: download model weights if not cached
        logger.info("[Init] DeepFace loaded successfully!")
    return deepface


def decode_image(data):
    """Decode a base64 image string or raw bytes into a PIL Image."""
    if isinstance(data, str):
        # Strip data URI prefix if present
        if "," in data:
            data = data.split(",", 1)[1]
        raw = base64.b64decode(data)
    else:
        raw = data
    return PILImage.open(io.BytesIO(raw)).convert("RGB")


def pil_to_numpy(pil_img):
    return np.array(pil_img)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": MODEL_NAME, "detector": DETECTOR})


@app.route("/extract", methods=["POST"])
def extract():
    """
    Extract a face embedding from an image.

    Accepts:
      - JSON body: { "image": "<base64_string>" }
      - OR multipart form: file field 'image'

    Returns:
      { "embedding": [128 or 512 floats], "face_count": int }
    """
    try:
        df = load_deepface()

        # --- Read image ---
        if request.is_json:
            body = request.get_json()
            image_data = body.get("image")
            if not image_data:
                return jsonify({"error": "No image provided"}), 400
            pil_img = decode_image(image_data)
        elif "image" in request.files:
            file = request.files["image"]
            pil_img = PILImage.open(file.stream).convert("RGB")
        else:
            return jsonify({"error": "No image provided. Send JSON {image: base64} or multipart form."}), 400

        img_array = pil_to_numpy(pil_img)

        # --- Extract embedding ---
        # RetinaFace + ArcFace + Cosine Distance = 99%+ Accuracy
        # ArcFace produces 512-D L2-normalized embeddings.
        embeddings = df.represent(
            img_path=img_array,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR,
            enforce_detection=True,   # Raise error if no face detected
            align=True,               # Critical for ArcFace accuracy
            normalization="base"      # Correct for ArcFace (already normalized)
        )

        if not embeddings:
            return jsonify({"error": "NO_FACE_DETECTED", "message": "No face found in the image."}), 422

        # Return first (primary) face embedding
        primary = embeddings[0]
        embedding = primary["embedding"]
        face_box = primary.get("facial_area", {})

        logger.info(f"[Extract] Extracted embedding. Faces found: {len(embeddings)}")

        return jsonify({
            "embedding": embedding,
            "embedding_dim": len(embedding),
            "face_count": len(embeddings),
            "face_area": face_box,
            "model": MODEL_NAME
        })

    except ValueError as e:
        # DeepFace raises ValueError when no face is detected
        error_str = str(e).lower()
        if "face" in error_str or "detect" in error_str:
            return jsonify({
                "error": "NO_FACE_DETECTED",
                "message": "We couldn't detect a face. Try better lighting or a closer shot."
            }), 422
        return jsonify({"error": str(e)}), 400

    except Exception as e:
        logger.error(f"[Extract] Error: {str(e)}")
        # Return error details to help debug 500s
        return jsonify({
            "error": "SERVER_ERROR",
            "message": str(e)
        }), 500


@app.route("/compare", methods=["POST"])
def compare():
    """
    Compare two face embeddings using cosine similarity.

    Accepts JSON body:
      { "embedding1": [...], "embedding2": [...] }

    Returns:
      { "similarity": float (0 to 1), "distance": float, "is_match": bool }
    """
    try:
        body = request.get_json()
        if not body:
            return jsonify({"error": "JSON body required"}), 400

        emb1 = body.get("embedding1")
        emb2 = body.get("embedding2")

        if emb1 is None or emb2 is None:
            return jsonify({"error": "Both embedding1 and embedding2 are required"}), 400

        e1 = np.array(emb1, dtype=np.float32)
        e2 = np.array(emb2, dtype=np.float32)

        # Cosine similarity = dot / (|e1| * |e2|)
        dot = np.dot(e1, e2)
        mag1 = np.linalg.norm(e1)
        mag2 = np.linalg.norm(e2)

        if mag1 == 0 or mag2 == 0:
            return jsonify({"similarity": 0.0, "distance": 1.0, "is_match": False})

        cosine_sim = float(dot / (mag1 * mag2))
        cosine_dist = 1.0 - cosine_sim

        # ArcFace threshold: distance < 0.68 means same person
        # Equivalent similarity: > 0.32
        # But with our cosine_sim (0 to 1 scale), use > 0.55 for match
        is_match = cosine_dist < 0.55

        logger.info(f"[Compare] Similarity: {cosine_sim:.4f}, Distance: {cosine_dist:.4f}, Match: {is_match}")

        return jsonify({
            "similarity": round(cosine_sim, 4),
            "distance": round(cosine_dist, 4),
            "is_match": is_match,
            "threshold": 0.55
        })

    except Exception as e:
        logger.error(f"[Compare] Error: {e}")
        return jsonify({"error": str(e)}), 500



@app.route("/health", methods=["GET"])
def health():
    """Simple health check endpoint."""
    return jsonify({
        "status": "online",
        "model": MODEL_NAME,
        "detector": DETECTOR
    })


if __name__ == "__main__":
    # Azure App Service sets the PORT environment variable
    port = int(os.environ.get("PORT", 5001))
    logger.info(f"[Startup] PixLand Face Service starting on port {port}")
    logger.info(f"[Startup] Model: {MODEL_NAME}, Detector: {DETECTOR}")
    app.run(host="0.0.0.0", port=port, debug=False)
