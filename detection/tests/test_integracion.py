"""
Práctica 4 - Construcción de Software [CONSOF]
Sesión 1: Pruebas de integración

Pruebas de integración con Django TestCase y el cliente de pruebas:
- Acceso a las rutas principales de la aplicación
- Respuesta correcta de las vistas (código 200 y plantilla usada)
- Flujo principal del sistema: iniciar sesión de detección,
  registrar emociones y detener la sesión.

Ejecución:
    python manage.py test detection.tests.test_integracion
"""

from unittest.mock import patch

from django.test import TestCase
from django.urls import reverse

from detection.models import DetectionSession, EmotionRecord


class PruebasRutasPrincipales(TestCase):
    """Verifica el acceso a las rutas principales del sistema."""

    def test_ruta_index_responde_200(self):
        respuesta = self.client.get(reverse("detection:index"))
        self.assertEqual(respuesta.status_code, 200)

    def test_ruta_historial_responde_200(self):
        respuesta = self.client.get(reverse("detection:historial"))
        self.assertEqual(respuesta.status_code, 200)

    @patch("detection.views.generar_frames")
    def test_ruta_video_feed_responde_como_stream(self, mock_frames):
        # Se simula el generador de frames para no depender de la cámara
        mock_frames.return_value = iter([b"--frame\r\n"])

        respuesta = self.client.get(reverse("detection:video_feed"))

        self.assertEqual(respuesta.status_code, 200)
        self.assertIn("multipart/x-mixed-replace", respuesta["Content-Type"])


class PruebasVistas(TestCase):
    """Verifica que las vistas respondan con las plantillas correctas."""

    def test_index_usa_plantilla_correcta(self):
        respuesta = self.client.get(reverse("detection:index"))
        self.assertTemplateUsed(respuesta, "detection/index.html")

    def test_historial_usa_plantilla_correcta(self):
        respuesta = self.client.get(reverse("detection:historial"))
        self.assertTemplateUsed(respuesta, "detection/historial.html")

    def test_historial_recibe_contexto_esperado(self):
        respuesta = self.client.get(reverse("detection:historial"))
        self.assertIn("sesiones", respuesta.context)
        self.assertIn("registros", respuesta.context)
        self.assertIn("emociones_count", respuesta.context)


class PruebasFlujoPrincipal(TestCase):
    """Verifica el flujo completo: iniciar sesión → registrar → detener."""

    def test_iniciar_sesion_crea_registro_en_bd(self):
        respuesta = self.client.post(reverse("detection:iniciar_sesion"))

        self.assertEqual(respuesta.status_code, 200)
        datos = respuesta.json()
        self.assertIn("session_id", datos)
        self.assertEqual(DetectionSession.objects.count(), 1)

    def test_detener_sesion_marca_fecha_de_fin(self):
        # 1. Se inicia una sesión de detección
        respuesta = self.client.post(reverse("detection:iniciar_sesion"))
        session_id = respuesta.json()["session_id"]

        # 2. Se detiene la sesión
        respuesta = self.client.post(
            reverse("detection:detener_sesion", args=[session_id])
        )

        self.assertEqual(respuesta.status_code, 200)
        sesion = DetectionSession.objects.get(id=session_id)
        self.assertIsNotNone(sesion.ended_at)

    def test_detener_sesion_inexistente_devuelve_404(self):
        respuesta = self.client.post(
            reverse("detection:detener_sesion", args=[9999])
        )
        self.assertEqual(respuesta.status_code, 404)

    def test_flujo_completo_con_registro_de_emociones(self):
        # 1. Iniciar sesión de detección
        session_id = self.client.post(
            reverse("detection:iniciar_sesion")
        ).json()["session_id"]
        sesion = DetectionSession.objects.get(id=session_id)

        # 2. Simular que la cámara registró dos emociones
        EmotionRecord.objects.create(
            session=sesion, emotion="happy", confidence=95.20,
            face_bbox=[50, 40, 80, 80],
        )
        EmotionRecord.objects.create(
            session=sesion, emotion="neutral", confidence=80.10,
            face_bbox=[52, 41, 78, 79],
        )

        # 3. Detener la sesión
        self.client.post(reverse("detection:detener_sesion", args=[session_id]))

        # 4. Verificar que el historial refleja los datos guardados
        respuesta = self.client.get(reverse("detection:historial"))
        self.assertEqual(respuesta.status_code, 200)
        self.assertEqual(len(respuesta.context["registros"]), 2)
        self.assertEqual(respuesta.context["emociones_count"].get("happy"), 1)
        self.assertEqual(respuesta.context["emociones_count"].get("neutral"), 1)
