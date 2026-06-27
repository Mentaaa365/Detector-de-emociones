from django.urls import path
from detection import views

app_name = "detection"

urlpatterns = [
    path("", views.index, name="index"),
    path("video_feed/", views.video_feed, name="video_feed"),
    path("historial/", views.historial, name="historial"),
    path("api/iniciar-sesion/", views.iniciar_sesion, name="iniciar_sesion"),
    path("api/detener-sesion/<int:session_id>/", views.detener_sesion, name="detener_sesion"),
]
