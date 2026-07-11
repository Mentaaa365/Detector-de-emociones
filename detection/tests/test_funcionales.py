"""
Práctica 4 - Construcción de Software [CONSOF]
Sesión 2: Pruebas funcionales

Pruebas funcionales de los casos de uso clave del sistema:
- Inicio del sistema (página principal carga con sus elementos)
- Funcionamiento de la detección en video (pipeline de frames)
- Visualización de resultados en la interfaz (historial de emociones)

Ejecución:
    python manage.py test detection.tests.test_funcionales
"""

from unittest.mock import patch

import numpy as np
from django.test import TestCase
from django.urls import reverse

from detection.models import DetectionSession, EmotionRecord
from detection.utils.camera import generar_frames


class PruebaInicioDelSistema(TestCase):
    """Caso de uso: el usuario abre el sistema en el navegador."""

    def test_pagina_principal_carga_correctamente(self):
        respuesta = self.client.get("/")

        self.assertEqual(respuesta.status_code, 200)
        self.assertTemplateUsed(respuesta, "detection/index.html")

    def test_pagina_principal_muestra_elementos_clave(self):
        respuesta = self.client.get("/")
        contenido = respuesta.content.decode("utf-8").lower()

        # La interfaz debe incluir el video y el acceso al historial
        self.assertIn("video_feed", contenido)
        self.assertIn("historial", contenido)


class PruebaDeteccionEnVideo(TestCase):
    """Caso de uso: el sistema procesa frames de la cámara y detecta emociones."""

    @patch("detection.utils.camera.analizar_emocion")
    @patch("detection.utils.camera.detectar_rostro")
    @patch("detection.utils.camera.cv2.VideoCapture")
    def test_pipeline_genera_frames_mjpeg_con_emocion(
        self, mock_capture, mock_rostro, mock_emocion
    ):
        # Se simula la cámara devolviendo un solo frame y luego cerrando
        frame_falso = np.zeros((240, 320, 3), dtype=np.uint8)
        camara = mock_capture.return_value
        camara.isOpened.return_value = True
        camara.read.side_effect = [(True, frame_falso), (False, None)]

        # Se simula un rostro detectado y una emoción con alta confianza
        mock_rostro.return_value = (frame_falso[40:120, 50:130], [50, 40, 80, 80])
        mock_emocion.return_value = {
            "emotion": "happy", "confidence": 96.30, "error": None,
        }

        frames = list(generar_frames())

        # El generador debe producir al menos un frame en formato MJPEG
        self.assertGreater(len(frames), 0)
        self.assertIn(b"--frame", frames[0])
        self.assertIn(b"Content-Type: image/jpeg", frames[0])
        mock_rostro.assert_called()
        mock_emocion.assert_called()

    @patch("detection.utils.camera.analizar_emocion")
    @patch("detection.utils.camera.detectar_rostro")
    @patch("detection.utils.camera.cv2.VideoCapture")
    def test_deteccion_guarda_emocion_en_sesion_activa(
        self, mock_capture, mock_rostro, mock_emocion
    ):
        # Sesión de detección activa
        sesion = DetectionSession.objects.create()

        frame_falso = np.zeros((240, 320, 3), dtype=np.uint8)
        camara = mock_capture.return_value
        camara.isOpened.return_value = True
        camara.read.side_effect = [(True, frame_falso), (False, None)]

        mock_rostro.return_value = (frame_falso[40:120, 50:130], [50, 40, 80, 80])
        mock_emocion.return_value = {
            "emotion": "surprise", "confidence": 91.75, "error": None,
        }

        list(generar_frames(session_id=sesion.id))

        # La emoción detectada debe quedar registrada en la base de datos
        registro = EmotionRecord.objects.filter(session=sesion).first()
        self.assertIsNotNone(registro)
        self.assertEqual(registro.emotion, "surprise")
        self.assertEqual(float(registro.confidence), 91.75)


class PruebaVisualizacionResultados(TestCase):
    """Caso de uso: el usuario consulta los resultados en la interfaz."""

    def setUp(self):
        # Datos de ejemplo: una sesión con tres emociones detectadas
        self.sesion = DetectionSession.objects.create()
        EmotionRecord.objects.create(
            session=self.sesion, emotion="happy", confidence=95.00)
        EmotionRecord.objects.create(
            session=self.sesion, emotion="happy", confidence=90.50)
        EmotionRecord.objects.create(
            session=self.sesion, emotion="sad", confidence=87.30)

    def test_historial_muestra_emociones_detectadas(self):
        respuesta = self.client.get(reverse("detection:historial"))
        contenido = respuesta.content.decode("utf-8").lower()

        self.assertEqual(respuesta.status_code, 200)
        self.assertIn("happy", contenido)
        self.assertIn("sad", contenido)

    def test_historial_cuenta_emociones_correctamente(self):
        respuesta = self.client.get(reverse("detection:historial"))
        conteo = respuesta.context["emociones_count"]

        self.assertEqual(conteo.get("happy"), 2)
        self.assertEqual(conteo.get("sad"), 1)

    def test_historial_lista_las_sesiones(self):
        respuesta = self.client.get(reverse("detection:historial"))
        self.assertEqual(len(respuesta.context["sesiones"]), 1)
