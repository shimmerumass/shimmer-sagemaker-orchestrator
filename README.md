# Touch Detection Random Forest Model

A machine learning-based touch detection system using Shimmer sensor data with UWB (Ultra-Wideband) distance measurements and accelerometer features.

## Overview

This project implements a touch detection pipeline that processes sensor data from left and right hand sensors to identify and classify touch events. The system uses a Random Forest classifier trained on extracted features from UWB distance data and accelerometer readings.

## Features

- **Multi-sensor Processing**: Handles left and right hand sensor data simultaneously
- **Event Detection**: Groups UWB events by time and filters noisy data
- **Feature Extraction**: Extracts statistical and frequency domain features from accelerometer data
- **Machine Learning Classification**: Uses trained Random Forest model for touch prediction
- **Flexible Output**: Supports both JSON and MATLAB (.mat) file outputs

## Project Structure

```
touch-detection-deploy/
├── code/
│   ├── __init__.py              # Python package initialization
│   ├── inference.py             # Main inference pipeline (SageMaker compatible)
│   ├── event_processor.py       # Event processing and feature extraction
│   └── requirements.txt         # Additional dependencies
├── model/
│   └── py_random_forest.joblib  # Trained Random Forest model
├── test.py                      # Test script with sample data
├── requirements.txt             # Main project dependencies
├── DC95_left-AllChannels.json   # Sample left hand sensor data
├── E169_right-AllChannels.json  # Sample right hand sensor data
└── README.md                    # This file
```

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/shimmerumass/shimmer-sagemaker-orchestrator.git
   cd touch-detection-deploy
   ```

2. **Create and activate virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On macOS/Linux
   # or
   venv\Scripts\activate     # On Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic Testing

Run the test script to process sample sensor data:

```bash
python test.py
```

This will:
- Load sample left and right hand sensor data
- Process the data through the complete pipeline
- Output results as JSON to console
- Save results as `touch_detection_results.mat`

### Input Data Format

The system expects JSON files with sensor data containing:
- `sampleRate`: Sampling frequency (Hz)
- `uwbDis`: UWB distance measurements
- `tagId`: Tag identifiers for UWB events
- `timestampCal`: Calibrated timestamps
- `Accel_WR_X_cal`, `Accel_WR_Y_cal`, `Accel_WR_Z_cal`: Calibrated accelerometer data

### Pipeline Components

1. **Event Grouping**: Groups UWB events by time proximity
2. **Tag Filtering**: Filters events by dominant tag
3. **Noise Filtering**: Removes noisy events based on distance thresholds
4. **Debounce Filtering**: Eliminates consecutive duplicate events
5. **Outlier Removal**: Uses MAD (Median Absolute Deviation) for outlier detection
6. **Feature Extraction**: Extracts statistical features from accelerometer data
7. **Classification**: Applies trained Random Forest model
8. **Touch Counting**: Aggregates results into touch counts

### Output Format

The system outputs:
```json
{
  "counts": {
    "left": 0,
    "right": 0
  },
  "leftEvents": [],
  "rightEvents": []
}
```

## Dependencies

- `numpy`: Numerical computations
- `scipy`: Signal processing and file I/O
- `pandas`: Data manipulation
- `scikit-learn`: Machine learning model
- `joblib`: Model serialization

## SageMaker Compatibility

The inference pipeline is designed for AWS SageMaker deployment with standard entry points:
- `model_fn()`: Model loading
- `input_fn()`: Input preprocessing
- `predict_fn()`: Prediction logic
- `output_fn()`: Output formatting

## Troubleshooting

### Common Issues

1. **No touch events detected**: Check if filtering parameters match your sensor data scale
2. **File not found errors**: Ensure all data files and model files are in correct locations
3. **Empty feature extraction**: Verify accelerometer data is present and properly formatted

### Debug Output

The system provides detailed debug information showing:
- Number of events at each filtering stage
- Distance measurements for each event
- Feature extraction results

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

This project is part of the Shimmer research initiative at UMass.

