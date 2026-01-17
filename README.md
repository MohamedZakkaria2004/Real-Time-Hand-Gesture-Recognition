# Requirements

* mediapipe 0.8.1
* OpenCV 3.4.2
* TensorFlow 2.3.0
* tf-nightly 2.5.0.dev
* scikit-learn 0.23.2
* matplotlib 3.3.2 

# Demo (Inference Test)

You can run a real-time inference demo using your webcam.

```bash
python app.py
```

**--device**


Camera device index (default: 0)

**--width**


Camera capture width (default: 960)

**--height**


Camera capture height (default: 540)

**--use_static_image_mode**


Enables MediaPipe’s static image mode (default: disabled)

**--min_detection_confidence**


Minimum hand detection confidence (default: 0.5)

**--min_tracking_confidence**


Minimum hand tracking confidence (default: 0.5)

# Project Directory Structure

```bash
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
**app.py**

A sample application for real-time inference.
This script:

* Performs hand sign recognition

* Performs finger gesture recognition

* Allows collecting training data for both models using the keyboard

**train_hand_sign_classifier.ipynb**

* A Jupyter Notebook used to train the hand sign recognition model using hand keypoint landmarks.

**train_finger_gesture_classifier.ipynb**

* A Jupyter Notebook used to train the finger gesture recognition model using fingertip coordinate history.

**model/hand_sign_classifier**

* This directory contains all files related to hand sign (static pose) recognition.

**hand_sign.csv**
Training dataset containing hand keypoint coordinates

**hand_sign_classifier.tflite**
* Trained TensorFlow Lite model

**hand_sign_labels.csv**
* Label definitions for hand sign classes

**hand_sign_classifier.py**
* Inference module for hand sign classification

**model/finger_gesture_classifier**

* This directory contains all files related to finger gesture (dynamic motion) recognition.

**finger_gesture.csv**
* Training dataset containing index fingertip coordinate history

**finger_gesture_classifier.tflite**
* Trained TensorFlow Lite model

**finger_gesture_labels.csv**
* Label definitions for finger gesture classes

**finger_gesture_classifier.py**
* Inference module for finger gesture classification

**utils/cvfpscalc.py**

* A utility module used to calculate and display FPS during inference.

