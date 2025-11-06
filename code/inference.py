import boto3
import json
import numpy as np
from joblib import load
from event_processor import (
    group_uwb_events_by_time, filter_by_dominant_tag, filter_noisy_events,
    filter_consecutive, remove_outliers_mad, extract_features, attach_predictions, count_touches
)
import os

# Initialize S3 client once
s3 = boto3.client("s3")

def load_json_from_s3(s3_uri):
    # s3_uri format: 's3://bucket/key.json'
    parts = s3_uri.replace("s3://", "").split("/", 1)
    bucket, key = parts[0], parts[1]
    response = s3.get_object(Bucket=bucket, Key=key)
    return json.loads(response['Body'].read())

def model_fn(model_dir):
    return load(os.path.join(model_dir, "model", "py_random_forest.joblib"))

def input_fn(request_body, content_type):
    data = json.loads(request_body)
    # Support S3 URI input (for SageMaker endpoint)
    if 'left_s3_uri' in data and 'right_s3_uri' in data:
        left = load_json_from_s3(data['left_s3_uri'])
        right = load_json_from_s3(data['right_s3_uri'])
        return {"leftSensorData": left, "rightSensorData": right}
    elif 'leftSensorData' in data and 'rightSensorData' in data:
        # Local test, merged dict
        return data
    else:
        raise ValueError("Input must contain left/right S3 URIs or sensor data blocks.")

def predict_fn(inputs, model):
    Fs = inputs['leftSensorData']['sampleRate']
    # --- Left hand ---
    left = group_uwb_events_by_time(inputs['leftSensorData'], Fs)
    print(f"Left events after grouping: {len(left)}")
    left = filter_by_dominant_tag(left)
    print(f"Left events after filter_by_dominant_tag: {len(left)}")
    if left:
        for i, event in enumerate(left):
            distances = event['uwbData']
            print(f"Event {i}: {len(distances)} points, init_dist: {np.mean(distances[:min(3, len(distances))]):.1f}")
    left = filter_noisy_events(left)
    print(f"Left events after filter_noisy_events: {len(left)}")
    left = filter_consecutive(left)
    print(f"Left events after filter_consecutive: {len(left)}")
    left = remove_outliers_mad(left)
    print(f"Left events after remove_outliers_mad: {len(left)}")
    X_left = extract_features(left, Fs)
    print(f"Left features shape: {X_left.shape}")
    print(f"Left features columns: {list(X_left.columns) if hasattr(X_left, 'columns') else 'No columns'}")
    if X_left.empty:
        print("Warning: X_left is empty!")
        left_preds = []
    else:
        left_preds = model.predict(X_left)
    left = attach_predictions(left, left_preds)
    # --- Right hand ---
    right = group_uwb_events_by_time(inputs['rightSensorData'], Fs)
    print(f"Right events after grouping: {len(right)}")
    right = filter_by_dominant_tag(right)
    print(f"Right events after filter_by_dominant_tag: {len(right)}")
    if right:
        for i, event in enumerate(right):
            distances = event['uwbData']
            print(f"Right Event {i}: {len(distances)} points, init_dist: {np.mean(distances[:min(3, len(distances))]):.1f}")
    right = filter_noisy_events(right)
    print(f"Right events after filter_noisy_events: {len(right)}")
    right = filter_consecutive(right)
    print(f"Right events after filter_consecutive: {len(right)}")
    right = remove_outliers_mad(right)
    print(f"Right events after remove_outliers_mad: {len(right)}")
    X_right = extract_features(right, Fs)
    print(f"Right features shape: {X_right.shape}")
    print(f"Right features columns: {list(X_right.columns) if hasattr(X_right, 'columns') else 'No columns'}")
    if X_right.empty:
        print("Warning: X_right is empty!")
        right_preds = []
    else:
        right_preds = model.predict(X_right)
    right = attach_predictions(right, right_preds)
    # Event counting
    counts = count_touches(left, right)
    return { 'counts': counts, 'leftEvents': left, 'rightEvents': right }

def output_fn(prediction, accept):
    return json.dumps(prediction)
