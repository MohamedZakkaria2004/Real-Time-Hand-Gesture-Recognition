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

# Training

Both hand sign recognition and finger gesture recognition support adding new training data and retraining the models.

**Hand Sign Recognition Training**

1. Learning Data Collection

Press k to enter hand sign data collection mode
(displayed as MODE: Logging Key Point).

While in this mode, press 0 to 9 to save hand keypoint data to:

```bash
model/hand_sign_classifier/hand_sign.csv
```
<img width="1632" height="272" alt="102345725-28d26280-3fe1-11eb-9eeb-8c938e3f625b" src="https://github.com/user-attachments/assets/83c63da4-ed04-4bba-b06a-0a73c1bc2680" />

**CSV format**

* 1st column: Pressed number (used as class ID)

* Remaining columns: Hand keypoint coordinates

By default, the following classes are included:

* 0: Open hand

* 1: Closed hand

* 2: Pointing

You can add additional classes (IDs 3 and above) or delete existing rows to rebuild the dataset.

**2. Model Training**

Open the notebook below and run all cells:
```bash
train_hand_sign_classifier.ipynb
```

To change the number of classes:

Update:
```bash
NUM_CLASSES = 3
```

Edit labels in:
```bash
model/hand_sign_classifier/hand_sign_labels.csv
```

**Key Point Coordinates**
<img width="1543" height="538" alt="102242918-ed328c80-3f3d-11eb-907c-61ba05678d54" src="https://github.com/user-attachments/assets/9c106ddb-71f6-45b3-90e7-d2122baf0e48" />

# Finger Gesture Recognition Training

**1. Learning Data Collection**

Press h to enter finger gesture data collection mode
(displayed as MODE: Logging Point History).

While in this mode, press 0 to 9 to save fingertip coordinate history to:
```bash
model/finger_gesture_classifier/finger_gesture.csv
```
<img width="1628" height="272" alt="102345850-54ede380-3fe1-11eb-8d04-88e351445898" src="https://github.com/user-attachments/assets/fe1b5947-f5cf-41ba-8b89-67644b7198f2" />

**CSV format**

* 1st column: Pressed number (used as class ID)

* Remaining columns: Index fingertip coordinate history

By default, the following classes are included:

* 0: Stationary

* 1: Clockwise

* 2: Counterclockwise

* 3: Moving

**2. Model Training**

Open the notebook below and run all cells:
```bash
train_finger_gesture_classifier.ipynb
```

To change the number of classes:

Update:
```bash
NUM_CLASSES = 4
```

Edit labels in:
```bash
model/finger_gesture_classifier/finger_gesture_labels.csv
```

