from unittest.mock import patch

from django.test import TestCase

from detection.utils.emotion_analyzer import analizar_emocion


class EmotionAnalyzerTests(TestCase):
    """Pruebas de detection/utils/emotion_analyzer.py usando DeepFace mockeado."""

    @patch("detection.utils.emotion_analyzer.DeepFace.analyze")
    def test_rostro_detectado_devuelve_emocion_dominante(self, mock_analyze):
        mock_analyze.return_value = [
            {
                "dominant_emotion": "happy",
                "emotion": {"happy": 98.4321, "sad": 1.5679},
            }
        ]

        resultado = analizar_emocion(frame="frame_falso")

        self.assertEqual(resultado["emotion"], "happy")
        self.assertEqual(resultado["confidence"], 98.43)
        self.assertIsNone(resultado["error"])

    @patch("detection.utils.emotion_analyzer.DeepFace.analyze")
    def test_sin_rostro_devuelve_error_controlado(self, mock_analyze):
        mock_analyze.side_effect = ValueError("Face could not be detected")

        resultado = analizar_emocion(frame="frame_falso")

        self.assertIsNone(resultado["emotion"])
        self.assertEqual(resultado["confidence"], 0.0)
        self.assertEqual(resultado["error"], "Rostro no detectado")
