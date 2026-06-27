"""
Script manual para probar detection/utils/emotion_analyzer.py con una foto
real, sin depender de camera.py ni face_detector.py.

Uso:
    python scripts/probar_emotion_analyzer.py ruta/a/una/foto_con_rostro.jpg
"""
import sys
from pathlib import Path

import cv2

sys.path.append(str(Path(__file__).resolve().parent.parent))

from detection.utils.emotion_analyzer import analizar_emocion


def main():
    if len(sys.argv) != 2:
        print("Uso: python scripts/probar_emotion_analyzer.py <ruta_imagen>")
        sys.exit(1)

    ruta_imagen = sys.argv[1]
    frame = cv2.imread(ruta_imagen)

    if frame is None:
        print(f"No se pudo leer la imagen: {ruta_imagen}")
        sys.exit(1)

    resultado = analizar_emocion(frame)
    print(resultado)


if __name__ == "__main__":
    main()
