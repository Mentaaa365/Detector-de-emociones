from django.db import models
from django.conf import settings

# Opciones de emociones permitidas en el sistema (choices para EmotionRecord)
EMOTION_CHOICES = [
    ('happy', 'Feliz (happy)'),
    ('sad', 'Triste (sad)'),
    ('angry', 'Enojado (angry)'),
    ('surprise', 'Sorprendido (surprise)'),
    ('neutral', 'Neutral (neutral)'),
    ('fear', 'Temeroso (fear)'),
    ('disgust', 'Disgustado (disgust)'),
]

class DetectionSession(models.Model):
    """
    Representa una sesión de detección de emociones faciales en tiempo real.
    Agrupa múltiples capturas o registros individuales de emociones durante un periodo de tiempo.
    """
    # Enlace al usuario de Django (opcional, null=True/blank=True si no ha iniciado sesión)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        verbose_name="Usuario"
    )
    # Fecha y hora exactas del inicio de la sesión de detección
    started_at = models.DateTimeField(
        auto_now_add=True,
        verbose_name="Inicio de Sesión"
    )
    # Fecha y hora exactas del fin de la sesión (puede registrarse al cerrar el stream)
    ended_at = models.DateTimeField(
        null=True,
        blank=True,
        verbose_name="Fin de Sesión"
    )
    # Información del dispositivo o navegador desde donde se origina la transmisión
    device_info = models.CharField(
        max_length=255,
        null=True,
        blank=True,
        verbose_name="Información del Dispositivo"
    )

    class Meta:
        verbose_name = "Sesión de Detección"
        verbose_name_plural = "Sesiones de Detección"

    def __str__(self):
        # Muestra el ID de sesión y la fecha de inicio en la representación textual
        return f"Sesión {self.id} (Iniciada: {self.started_at})"


class EmotionRecord(models.Model):
    """
    Representa un registro individual de emoción facial detectada en un frame de video.
    Pertenece a una sesión de detección específica.
    """
    # Relación muchos a uno con DetectionSession. Si la sesión se borra, se eliminan sus registros (CASCADE)
    session = models.ForeignKey(
        DetectionSession,
        on_delete=models.CASCADE,
        related_name='records',
        verbose_name="Sesión de Detección"
    )
    # La emoción detectada elegida de entre las opciones de EMOTION_CHOICES
    emotion = models.CharField(
        max_length=20,
        choices=EMOTION_CHOICES,
        verbose_name="Emoción"
    )
    # Nivel de confianza o probabilidad de la emoción expresada en porcentaje (ej: 98.50)
    confidence = models.DecimalField(
        max_digits=5,
        decimal_places=2,
        verbose_name="Confianza (%)"
    )
    # Fecha y hora en que se procesó el frame y se grabó la detección
    detected_at = models.DateTimeField(
        auto_now_add=True,
        verbose_name="Detectado el"
    )
    # Campo JSON para almacenar las coordenadas del rectángulo contenedor del rostro [x, y, w, h]
    face_bbox = models.JSONField(
        null=True,
        blank=True,
        verbose_name="Caja Delimitadora de Rostro (BBox)"
    )

    class Meta:
        verbose_name = "Registro de Emoción"
        verbose_name_plural = "Registros de Emociones"

    def __str__(self):
        # Muestra la emoción y la fecha de detección en la representación textual
        return f"{self.emotion} - {self.detected_at}"
