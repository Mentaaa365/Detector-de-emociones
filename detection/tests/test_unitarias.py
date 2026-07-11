"""
Práctica 4 - Construcción de Software [CONSOF]
Sesión 1: Pruebas unitarias

Pruebas unitarias sobre las funciones críticas del sistema:
- Detector de emociones (detection/utils/emotion_analyzer.py)
- Detector de rostro (detection/utils/face_detector.py)
- Carga de las funciones principales del sistema
- Manejo de errores simples

Se usa unittest.mock para simular DeepFace y el clasificador de OpenCV,
de modo que las pruebas se ejecuten rápido y sin cámara ni modelos reales.

Ejecución:
    python manage.py test detection.tests.test_unitarias
"""

from unittest.mock import MagicMock, patch

import numpy as np
from django.test import SimpleTestCase

from detection.utils import face_detector
from detection.utils.emotion_analyzer import analizar_emocion
from detection.utils.face_detector import detectar_rostro


class PruebasDetectorEmociones(SimpleTestCase):
    """Verifica que el detector de emociones funcione correctamente."""

    @patch("detection.utils.emotion_analyzer.DeepFace.analyze")
    def test_detecta_emocion_dominante(self, mock_analyze):
        # Se simula la respuesta de DeepFace para un rostro feliz
        mock_analyze.return_value = [
            {
                "dominant_emotion": "happy",
                "emotion": {"happy": 97.5312, "sad": 1.2, "neutral": 1.2688},
            }
        ]

        resultado = analizar_emocion(frame="frame_de_prueba")

        self.assertEqual(resultado["emotion"], "happy")
        self.assertEqual(resultado["confidence"], 97.53)
        self.assertIsNone(resultado["error"])

    @patch("detection.utils.emotion_analyzer.DeepFace.analyze")
    def test_detecta_otras_emociones(self, mock_analyze):
        # El sistema debe devolver cualquier emoción dominante, no solo "happy"
        mock_analyze.return_value = [
            {
                "dominant_emotion": "sad",
                "emotion": {"sad": 88.0, "happy": 12.0},
            }
        ]

        resultado = analizar_emocion(frame="frame_de_prueba")

        self.assertEqual(resultado["emotion"], "sad")
        self.assertEqual(resultado["confidence"], 88.0)


class PruebasDeteccionRostro(SimpleTestCase):
    """Verifica la función detectar_rostro() basada en Haar Cascade."""

    def test_imagen_sin_rostro_devuelve_none(self):
        # Una imagen completamente negra no contiene ningún rostro
        frame_vacio = np.zeros((240, 320, 3), dtype=np.uint8)

        resultado = detectar_rostro(frame_vacio)

        self.assertIsNone(resultado)

    def test_rostro_detectado_devuelve_recorte_y_bbox(self):
        # Se simula que el clasificador encuentra un rostro en (50, 40, 80, 80)
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        clasificador_falso = MagicMock()
        clasificador_falso.detectMultiScale.return_value = np.array(
            [[50, 40, 80, 80]]
        )

        with patch.object(face_detector, "_classifier", clasificador_falso):
            resultado = detectar_rostro(frame)

        self.assertIsNotNone(resultado)
        crop, bbox = resultado
        # El bbox debe ser una lista [x, y, w, h] de enteros
        self.assertEqual(bbox, [50, 40, 80, 80])
        # El recorte debe ser una imagen no vacía
        self.assertGreater(crop.size, 0)

    def test_selecciona_el_rostro_mas_grande(self):
        # Si hay varios rostros, se elige el de mayor área (más cercano a la cámara)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        rostros = np.array([[10, 10, 50, 50], [100, 100, 120, 120]])
        clasificador_falso = MagicMock()
        clasificador_falso.detectMultiScale.return_value = rostros

        with patch.object(face_detector, "_classifier", clasificador_falso):
            _, bbox = detectar_rostro(frame)

        self.assertEqual(bbox, [100, 100, 120, 120])


class PruebasCargaFuncionesPrincipales(SimpleTestCase):
    """Verifica que las funciones principales del sistema se carguen bien."""

    def test_funciones_importan_y_son_invocables(self):
        from detection.utils.camera import generar_frames
        from detection.utils.emotion_analyzer import analizar_emocion
        from detection.utils.face_detector import detectar_rostro

        self.assertTrue(callable(analizar_emocion))
        self.assertTrue(callable(detectar_rostro))
        self.assertTrue(callable(generar_frames))

    def test_clasificador_haar_cargado(self):
        # El clasificador de OpenCV no debe estar vacío al iniciar el módulo
        self.assertFalse(face_detector._classifier.empty())

    def test_vistas_principales_definidas(self):
        from detection import views

        self.assertTrue(callable(views.index))
        self.assertTrue(callable(views.video_feed))
        self.assertTrue(callable(views.historial))
        self.assertTrue(callable(views.iniciar_sesion))
        self.assertTrue(callable(views.detener_sesion))


class PruebasManejoErrores(SimpleTestCase):
    """Verifica el manejo de errores simples del detector de emociones."""

    @patch("detection.utils.emotion_analyzer.DeepFace.analyze")
    def test_sin_rostro_devuelve_error_controlado(self, mock_analyze):
        # DeepFace lanza ValueError cuando no encuentra un rostro
        mock_analyze.side_effect = ValueError("Face could not be detected")

        resultado = analizar_emocion(frame="frame_de_prueba")

        self.assertIsNone(resultado["emotion"])
        self.assertEqual(resultado["confidence"], 0.0)
        self.assertEqual(resultado["error"], "Rostro no detectado")

    @patch("detection.utils.emotion_analyzer.DeepFace.analyze")
    def test_error_inesperado_no_rompe_el_sistema(self, mock_analyze):
        # Cualquier otra excepción debe devolver un mensaje, no romper la app
        mock_analyze.side_effect = RuntimeError("Fallo interno del modelo")

        resultado = analizar_emocion(frame="frame_de_prueba")

        self.assertIsNone(resultado["emotion"])
        self.assertEqual(resultado["confidence"], 0.0)
        self.assertIn("Fallo interno", resultado["error"])
