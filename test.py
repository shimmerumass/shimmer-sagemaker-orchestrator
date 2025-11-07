import json
from joblib import load
from code.inference import predict_fn

with open('DC95_left-AllChannels.json', 'r') as f:
    left_data = json.load(f)
with open('E169_right-AllChannels.json', 'r') as f:
    right_data = json.load(f)

inputs = {
    "leftSensorData": left_data,
    "rightSensorData": right_data
}

model = load('model/py_random_forest.joblib')

try:
    result = predict_fn(inputs, model)
    print(json.dumps(result, indent=2))
except Exception as e:
    print(f"Error during prediction: {e}")
    print(f"Input keys: {list(inputs.keys())}")
    if 'leftSensorData' in inputs:
        print(f"Left sensor keys: {list(inputs['leftSensorData'].keys())}")
    if 'rightSensorData' in inputs:
        print(f"Right sensor keys: {list(inputs['rightSensorData'].keys())}")
