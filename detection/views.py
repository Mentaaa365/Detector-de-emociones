from django.http import StreamingHttpResponse, JsonResponse
from django.shortcuts import render
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

from detection.models import DetectionSession, EmotionRecord
from detection.utils.camera import generar_frames

_current_session_id = None


def index(request):
    return render(request, "detection/index.html")


def video_feed(request):
    return StreamingHttpResponse(
        generar_frames(session_id=_current_session_id),
        content_type="multipart/x-mixed-replace; boundary=frame",
    )


def historial(request):
    sesiones = DetectionSession.objects.order_by("-started_at")[:20]
    registros = EmotionRecord.objects.order_by("-detected_at")[:50]

    emociones_count = {}
    for r in EmotionRecord.objects.all():
        emociones_count[r.emotion] = emociones_count.get(r.emotion, 0) + 1

    return render(request, "detection/historial.html", {
        "sesiones": sesiones,
        "registros": registros,
        "emociones_count": emociones_count,
    })


@csrf_exempt
@require_POST
def iniciar_sesion(request):
    global _current_session_id
    device = request.META.get("HTTP_USER_AGENT", "Desconocido")
    sesion = DetectionSession.objects.create(
        user=request.user if request.user.is_authenticated else None,
        device_info=device,
    )
    _current_session_id = sesion.id
    return JsonResponse({"session_id": sesion.id, "started_at": str(sesion.started_at)})


@csrf_exempt
@require_POST
def detener_sesion(request, session_id):
    global _current_session_id
    try:
        sesion = DetectionSession.objects.get(id=session_id)
        sesion.ended_at = timezone.now()
        sesion.save()
        _current_session_id = None
        return JsonResponse({"status": "Sesion finalizada"})
    except DetectionSession.DoesNotExist:
        return JsonResponse({"error": "Sesion no encontrada"}, status=404)
