"""Fatma: deviceless eye tracker.

Webcam-only gaze tracking that drives the mouse cursor — no headset, no IR
illuminator, no dedicated hardware. Iris landmarks come from MediaPipe's
FaceLandmarker; a thin-plate-spline calibration maps them to screen space.

The package is split so that every piece of maths and state machinery is
importable and testable without a camera, a display or a mouse:

    fatma.filters      OneEuroFilter, TemporalMedian
    fatma.landmarks    iris / eyelid / brow feature extraction
    fatma.tps          thin-plate-spline interpolation
    fatma.calibration  Calibrator, calibration file I/O
    fatma.gestures     wink + brow-raise state machine (emits events)
    fatma.camera       threaded frame grabber
    fatma.pointer      mouse backends (Quartz, PyAutoGUI, Null)
    fatma.app          the OpenCV/MediaPipe runtime loop

Only ``fatma.app`` imports cv2/mediapipe/pyautogui.
"""

__all__ = ["__version__", "NAME", "LONG_NAME"]

__version__ = "0.2.0"
NAME = "Fatma"
LONG_NAME = "Fatma: deviceless eye tracker"
