import boto3
import json
import numpy as np
import os
import traceback
from joblib import load
from .event_processor import (
    group_uwb_events_by_time,
    filter_by_dominant_tag,
    filter_noisy_events,
    filter_consecutive,
    remove_outliers_mad,
    extract_features,
    attach_predictions,
    count_touches
)

# Initialize S3 client once (safe even if not used)
s3 = boto3.client("s3")


# ------------------------------
# Helpers
# ------------------------------
def load_json_from_s3(s3_uri):
    """Load JSON data from S3 URI."""
    parts = s3_uri.replace("s3://", "").split("/", 1)
    bucket, key = parts[0], parts[1]
    response = s3.get_object(Bucket=bucket, Key=key)
    return json.loads(response["Body"].read())


# ------------------------------
# SageMaker entry points
# ------------------------------
def model_fn(model_dir):
    """Load trained model from SageMaker model directory."""
    try:
        model_path = os.path.join(model_dir, "model", "py_random_forest.joblib")
        print(f"[INFO] Loading model from: {model_path}")
        model = load(model_path)
        print("[INFO] Model loaded successfully.")
        return model
    except Exception as e:
        print("[ERROR] model_fn failed:", traceback.format_exc())
        raise e


def input_fn(request_body, content_type):
    """Parse and validate input request."""
    try:
        print("[INFO] input_fn triggered.")
        data = json.loads(request_body)
        if "left_s3_uri" in data and "right_s3_uri" in data:
            left = load_json_from_s3(data["left_s3_uri"])
            right = load_json_from_s3(data["right_s3_uri"])
            return {"leftSensorData": left, "rightSensorData": right}
        elif "leftSensorData" in data and "rightSensorData" in data:
            return data
        else:
            raise ValueError("Input must contain left/right S3 URIs or sensor data blocks.")
    except Exception as e:
        print("[ERROR] input_fn failed:", traceback.format_exc())
        raise e


def predict_fn(inputs, model):
    """Run the model on processed input data."""
    try:
        print("[INFO] predict_fn triggered.")
        Fs = inputs["leftSensorData"]["sampleRate"]

        # --- LEFT HAND ---
        left = group_uwb_events_by_time(inputs["leftSensorData"], Fs)
        print(f"[DEBUG] Left grouped events: {len(left)}")
        left = filter_by_dominant_tag(left)
        left = filter_noisy_events(left)
        left = filter_consecutive(left)
        left = remove_outliers_mad(left)
        X_left = extract_features(left, Fs)
        left_preds = model.predict(X_left) if not X_left.empty else []
        left = attach_predictions(left, left_preds)

        # --- RIGHT HAND ---
        right = group_uwb_events_by_time(inputs["rightSensorData"], Fs)
        print(f"[DEBUG] Right grouped events: {len(right)}")
        right = filter_by_dominant_tag(right)
        right = filter_noisy_events(right)
        right = filter_consecutive(right)
        right = remove_outliers_mad(right)
        X_right = extract_features(right, Fs)
        right_preds = model.predict(X_right) if not X_right.empty else []
        right = attach_predictions(right, right_preds)

        # --- COMBINE RESULTS ---
        counts = count_touches(left, right)
        result = {"counts": counts, "leftEvents": left, "rightEvents": right}
        print("[INFO] Prediction completed successfully.")
        return result

    except Exception as e:
        print("[ERROR] predict_fn failed:", traceback.format_exc())
        raise e


def output_fn(prediction, accept):
    """Format model output as JSON."""
    try:
        print("[INFO] output_fn triggered.")
        return json.dumps(prediction)
    except Exception as e:
        print("[ERROR] output_fn failed:", traceback.format_exc())
        raise e
