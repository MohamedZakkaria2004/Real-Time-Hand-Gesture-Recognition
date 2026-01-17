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

--device
Camera device index (default: 0)

--width
Camera capture width (default: 960)

--height
Camera capture height (default: 540)

--use_static_image_mode
Enables MediaPipe’s static image mode (default: disabled)

--min_detection_confidence
Minimum hand detection confidence (default: 0.5)

--min_tracking_confidence
Minimum hand tracking confidence (default: 0.5)
