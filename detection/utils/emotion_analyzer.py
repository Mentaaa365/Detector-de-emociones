"""
Envuelve DeepFace.analyze() para devolver la emoción dominante de un frame
en un formato simple, listo para que camera.py lo use en el loop de captura.
"""
import logging

from deepface import DeepFace

logger = logging.getLogger(__name__)


def analizar_emocion(frame):
    """
    Analiza un frame (array BGR de OpenCV) y devuelve la emoción dominante.

    Args:
        frame: imagen como numpy array, tal como la entrega
            cv2.VideoCapture.read() o el recorte de rostro de face_detector.py.

    Returns:
        dict con:
            - "emotion" (str | None): una de las claves de EMOTION_CHOICES
              (detection/models.py), o None si no se detectó rostro.
            - "confidence" (float): confianza 0-100 de la emoción dominante.
            - "error" (str | None): mensaje legible si algo falló.
    """
    try:
        resultados = DeepFace.analyze(
            frame,
            actions=["emotion"],
            enforce_detection=False,
            detector_backend="skip",
            silent=True,
        )
    except ValueError:
        return {"emotion": None, "confidence": 0.0, "error": "Rostro no detectado"}
    except Exception as exc:
        logger.exception("Fallo inesperado al analizar la emoción del frame")
        return {"emotion": None, "confidence": 0.0, "error": str(exc)}

    # Con un solo rostro esperado en este proyecto, se usa el primero de la lista.
    resultado = resultados[0] if isinstance(resultados, list) else resultados

    emocion_dominante = resultado["dominant_emotion"]
    confianza = round(float(resultado["emotion"][emocion_dominante]), 2)

    return {"emotion": emocion_dominante, "confidence": confianza, "error": None}
