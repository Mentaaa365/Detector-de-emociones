"""
MJPEG frame generator for the emotion detection pipeline.

Exposes generar_frames(), a generator designed to be consumed directly by
Django's StreamingHttpResponse with content_type
'multipart/x-mixed-replace; boundary=frame' (wired up by Integrante 4).

Pipeline per frame:
    VideoCapture → face_detector.detectar_rostro()
                 → emotion_analyzer.analizar_emocion()  (only when face found)
                 → annotate original frame
                 → JPEG encode
                 → yield MJPEG part
"""

import cv2

from detection.utils.emotion_analyzer import analizar_emocion
from detection.utils.face_detector import detectar_rostro

_MJPEG_BOUNDARY = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
_BOX_COLOR = (0, 255, 0)    # green bounding box
_TEXT_COLOR = (0, 255, 0)   # green label text
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.7
_THICKNESS = 2


def _encode_jpeg(frame) -> bytes | None:
    """
    Encode a BGR frame to JPEG bytes.

    Args:
        frame: BGR numpy array.

    Returns:
        Raw JPEG bytes, or None if encoding fails.
    """
    ok, buffer = cv2.imencode(".jpg", frame)
    if not ok:
        return None
    return buffer.tobytes()


def _draw_bbox(frame, bbox: list[int]) -> None:
    """
    Draw a green rectangle on the frame in-place.

    Args:
        frame: BGR numpy array (mutated in-place).
        bbox:  [x, y, w, h] in pixels.
    """
    x, y, w, h = bbox
    cv2.rectangle(frame, (x, y), (x + w, y + h), _BOX_COLOR, _THICKNESS)


def _draw_label(frame, bbox: list[int], label: str) -> None:
    """
    Draw a text label just above the bounding box in-place.

    Clamps the y position so the label stays inside the frame when the face
    is near the top edge.

    Args:
        frame: BGR numpy array (mutated in-place).
        bbox:  [x, y, w, h] in pixels.
        label: text string to render.
    """
    x, y = bbox[0], bbox[1]
    text_y = max(y - 10, 20)
    cv2.putText(frame, label, (x, text_y), _FONT, _FONT_SCALE, _TEXT_COLOR, _THICKNESS, cv2.LINE_AA)


def generar_frames(source: int | str = 0, session_id: int | None = None):
    """
    Yield MJPEG parts from the camera with emotion overlay.
    If session_id is provided, saves each detection to the database.
    """
    import time
    cap = cv2.VideoCapture(source)
    last_save = 0
    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            detection = detectar_rostro(frame)

            if detection is not None:
                crop, bbox = detection
                _draw_bbox(frame, bbox)

                resultado = analizar_emocion(crop)
                if resultado["error"] is None:
                    label = f"{resultado['emotion']}  {resultado['confidence']:.1f}%"
                    _draw_label(frame, bbox, label)

                    now = time.time()
                    if session_id and now - last_save >= 2:
                        last_save = now
                        _guardar_registro(session_id, resultado, bbox)

            jpeg = _encode_jpeg(frame)
            if jpeg is None:
                continue

            yield _MJPEG_BOUNDARY + jpeg + b"\r\n"
    finally:
        cap.release()


def _guardar_registro(session_id, resultado, bbox):
    try:
        from detection.models import EmotionRecord
        EmotionRecord.objects.create(
            session_id=session_id,
            emotion=resultado["emotion"],
            confidence=resultado["confidence"],
            face_bbox=bbox,
        )
    except Exception:
        pass
