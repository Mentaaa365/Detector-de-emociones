"""
Face detection utilities using OpenCV's Haar Cascade classifier.

Provides a single public function, detectar_rostro(), that locates the most
prominent face in a BGR frame and returns the crop together with its bounding
box — the format expected by emotion_analyzer.analizar_emocion() and
EmotionRecord.face_bbox respectively.
"""

import cv2
import numpy as np

_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_classifier = cv2.CascadeClassifier(_CASCADE_PATH)


def detectar_rostro(frame: np.ndarray):
    """
    Detect the largest face in a BGR frame and return its crop and bounding box.

    Uses OpenCV's Haar Cascade (haarcascade_frontalface_default.xml) bundled
    with the opencv-python package — no external model files required.

    When multiple faces are present, the one with the largest area (w * h) is
    chosen, which is a reasonable proxy for "closest to the camera" and keeps
    the downstream emotion analysis focused on a single subject.

    Args:
        frame: BGR numpy array as produced by cv2.VideoCapture.read().
               Shape must be (H, W, 3).

    Returns:
        A tuple (crop, bbox) where:
            crop  — numpy array, BGR, shape (h, w, 3). Ready to pass directly
                    to analizar_emocion().
            bbox  — list [x, y, w, h] of Python ints. Compatible with
                    EmotionRecord.face_bbox (JSONField).
        None if no face is detected in the frame.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = _classifier.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(48, 48),
    )

    if len(faces) == 0:
        return None

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

    # Padding de 40% alrededor del rostro para que el modelo de emociones
    # tenga contexto de cejas, frente y barbilla
    pad_w = int(w * 0.4)
    pad_h = int(h * 0.4)
    fh, fw = frame.shape[:2]
    cx = max(x - pad_w, 0)
    cy = max(y - pad_h, 0)
    cx2 = min(x + w + pad_w, fw)
    cy2 = min(y + h + pad_h, fh)

    crop = frame[cy:cy2, cx:cx2]
    return crop, [int(x), int(y), int(w), int(h)]
