from django.contrib import admin
from .models import DetectionSession, EmotionRecord

# Configuración del panel de administración para DetectionSession
@admin.register(DetectionSession)
class DetectionSessionAdmin(admin.ModelAdmin):
    # Campos a mostrar en la lista de registros del Django Admin
    list_display = ('id', 'user', 'started_at', 'ended_at')
    # Permite ordenar por fecha de inicio y filtrar por usuario
    ordering = ('-started_at',)
    list_filter = ('started_at', 'user')

# Configuración del panel de administración para EmotionRecord
@admin.register(EmotionRecord)
class EmotionRecordAdmin(admin.ModelAdmin):
    # Campos a mostrar en la lista de registros
    list_display = ('id', 'session', 'emotion', 'confidence', 'detected_at')
    # Filtro lateral para buscar por emoción detectada y fecha
    list_filter = ('emotion', 'detected_at')
    # Ordenar por defecto por el registro más reciente
    ordering = ('-detected_at',)
