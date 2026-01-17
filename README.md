# Real-Time-Hand-Gesture-Recognition
This repository includes sample implementations and trained models for hand sign and finger gesture recognition. It provides TensorFlow Lite (TFLite) models for real-time inference, along with the corresponding training datasets and Jupyter notebooks used for model development.

# Requirements
* mediapipe 0.8.1
* OpenCV 3.4.2 
* Tensorflow 2.3.0 
* tf-nightly 2.5.0.dev
* scikit-learn 0.23.2
* matplotlib 3.3.2 or Later

# How to run the Demo

To run the demo using webcam, run this command:

```bash
python app.py
```

The below options can be specified when the demo is running:

--device
Camera device index to use (default: 0)

--width
Capture width (default: 960)

--height
Capture height (default: 540)

--use_static_image_mode
Enables MediaPipe’s static_image_mode (useful if you want frame-by-frame detection instead of tracking; default: off)

--min_detection_confidence
Minimum hand detection confidence threshold (default: 0.5)

--min_tracking_confidence
Minimum hand tracking confidence threshold (default: 0.5)
