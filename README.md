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

```text
--device
Camera device index to use (default: 0)
--width
Capture width (default: 960)
--height
Capture height (default: 540)
--use_static_image_mode
Enables MediaPipe’s static_image_mode (default: off)
--min_detection_confidence
Minimum hand detection confidence threshold (default: 0.5)
--min_tracking_confidence
Minimum hand tracking confidence threshold (default: 0.5)
```
# Project Directory

```text
│  app.py
│  train_hand_sign_classifier.ipynb
│  train_finger_gesture_classifier.ipynb
│
├─model
│  ├─hand_sign_classifier
│  │  │  hand_sign.csv
│  │  │  hand_sign_classifier.hdf5
│  │  │  hand_sign_classifier.py
│  │  │  hand_sign_classifier.tflite
│  │  └─ hand_sign_labels.csv
│  │
│  └─finger_gesture_classifier
│      │  finger_gesture.csv
│      │  finger_gesture_classifier.hdf5
│      │  finger_gesture_classifier.py
│      │  finger_gesture_classifier.tflite
│      └─ finger_gesture_labels.csv
│
└─utils
    └─cvfpscalc.py
```

# File & Directory Overview

_app.py_

* A sample application for real-time hand gesture inference using a webcam.
* This script performs hand sign recognition and finger gesture recognition, and also supports collecting training data:

* Hand sign recognition: keypoint-based landmark data

* Finger gesture recognition: index finger coordinate history over time

_train_hand_sign_classifier.ipynb_

* A Jupyter Notebook used to train the hand sign recognition model from keypoint landmark data.

_train_finger_gesture_classifier.ipynb_

* A Jupyter Notebook used to train the finger gesture recognition model from point-history (temporal motion) data.

**Model Directories**

_model/hand_sign_classifier_

* This directory contains all resources related to hand sign (static pose) recognition.

**Contents:**

_hand_sign.csv_
* Training dataset containing hand keypoint landmarks

_hand_sign_classifier.tflite_
* Trained TensorFlow Lite model for real-time inference

_hand_sign_labels.csv_
* Label definitions for hand sign classes

_hand_sign_classifier.py_
* Inference module for hand sign classification

_model/finger_gesture_classifier_

* This directory contains all resources related to finger gesture (dynamic motion) recognition.

**Contents:**

_finger_gesture.csv_
* Training dataset containing index finger coordinate history

_finger_gesture_classifier.tflite_
* Trained TensorFlow Lite model for real-time inference

_finger_gesture_labels.csv_
* Label definitions for finger gesture classes

_finger_gesture_classifier.py_
* Inference module for finger gesture classification

**Utilities**

_utils/cvfpscalc.py_
* A utility module used to calculate and display the current frames per second (FPS) during inference.
