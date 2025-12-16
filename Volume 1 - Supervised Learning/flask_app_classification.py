import uuid
import json
from datetime import datetime, timezone
from etl import etl_transform_for_pipeline 
from flask import Flask, request, jsonify
from pathlib import Path
import joblib
import pandas as pd

# Resolve base directory and models directory
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "saved_models"

# Fail fast if the models directory doesn't exist
if not MODELS_DIR.exists():
    raise FileNotFoundError(f"Models directory not found: {MODELS_DIR}")

# Define paths to the three saved model files
model_a_path = MODELS_DIR / "model_a.joblib"
model_b_path = MODELS_DIR / "model_b.joblib"
model_c_path = MODELS_DIR / "model_c.joblib"

# Ensure all model files exist
for p in [model_a_path, model_b_path, model_c_path]:
    if not p.exists():
        raise FileNotFoundError(f"Missing model file: {p}")

# Load the models (joblib will resolve custom code like etl.py if present in the same folder)
model_a = joblib.load(model_a_path)
model_b = joblib.load(model_b_path)
model_c = joblib.load(model_c_path)

# Create Flask app
app = Flask(__name__)


def utc_now_iso():
    """Return current UTC time in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


def make_request_id():
    """Generate a short unique request ID (e.g., REQ_a1b2c3d4e5f6)."""
    return f"REQ_{uuid.uuid4().hex[:12]}"


def payload_to_df(payload):
    """
    Convert incoming payload to a pandas DataFrame.
    Supports:
      - dict → single row
      - list of dicts → multiple rows
    """
    if isinstance(payload, dict):
        return pd.DataFrame([payload])
    if isinstance(payload, list):
        if len(payload) == 0:
            raise ValueError("Empty list received. Provide at least one row.")
        if not all(isinstance(x, dict) for x in payload):
            raise ValueError("List payload must be a list of objects (dict rows).")
        return pd.DataFrame(payload)
    raise ValueError("Payload must be an object or a list of objects.")


def proba_positive(model, X):
    """
    Extract probability of the positive class from predict_proba.
    Assumes binary classification with positive class at index 1.
    """
    if not hasattr(model, "predict_proba"):
        raise ValueError("Model does not support predict_proba().")
    p = model.predict_proba(X)
    if p.ndim != 2 or p.shape[1] < 2:
        raise ValueError("Unexpected predict_proba shape.")
    return p[:, 1]


def label_from_proba(p, threshold=0.5):
    """Convert probability to human-readable label."""
    return "High Risk" if p >= threshold else "Low Risk"


def store_result(record, out_path="predictions_log.jsonl"):
    """
    Append a prediction record to a JSONL file (one JSON object per line).
    Useful for audit trails and later analysis.
    """
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


@app.route("/health", methods=["GET"])
def health():
    """Health-check endpoint for monitoring / load balancers."""
    return jsonify({"status": "ok"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    Main prediction endpoint.
    
    Accepts two payload formats for flexibility:
      1. Direct features (object or array of objects)
      2. Wrapped format with optional metadata:
         {
           "customer_id": "...",
           "features_version": "...",
           "features": { ... }   # or array
         }
    """
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Invalid or missing JSON body."}), 400

    request_id = make_request_id()

    # Extract features and optional metadata
    if isinstance(data, dict) and "features" in data:
        customer_id = data.get("customer_id")
        features_version = data.get("features_version")
        raw_features = data["features"]
    else:
        customer_id = None
        features_version = None
        raw_features = data

    try:
        X = payload_to_df(raw_features)

        # Run inference on all three models
        pa = proba_positive(model_a, X)
        pb = proba_positive(model_b, X)
        pc = proba_positive(model_c, X)

        out = []
        for i in range(len(X)):
            record = {
                "request_id": request_id,
                "timestamp": utc_now_iso(),
                "customer_id": customer_id,
                "features_version": features_version,
                "features": X.iloc[i].to_dict(),
                "results": [
                    {
                        "model_id": "model_a",
                        "model_version": "1.0.0",
                        "probability": float(pa[i]),
                        "label": label_from_proba(float(pa[i])),
                    },
                    {
                        "model_id": "model_b",
                        "model_version": "1.0.0",
                        "probability": float(pb[i]),
                        "label": label_from_proba(float(pb[i])),
                    },
                    {
                        "model_id": "model_c",
                        "model_version": "1.0.0",
                        "probability": float(pc[i]),
                        "label": label_from_proba(float(pc[i])),
                    },
                ],
            }
            store_result(record)
            out.append(record)

        # Return single object if only one row, otherwise list
        return jsonify(out[0] if len(out) == 1 else out), 200

    except Exception as e:
        # Return error with request_id for traceability
        return jsonify({"request_id": request_id, "error": str(e)}), 400


if __name__ == "__main__":
    # Development server only — use gunicorn/uvicorn in production
    app.run(host="0.0.0.0", port=5000, debug=True)