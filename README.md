# Detector de Emociones Facial en Tiempo Real

Este proyecto es una aplicación web para la detección de emociones faciales en tiempo real usando Python, Django, OpenCV, DeepFace y PostgreSQL.

## Requerimientos del Sistema

* **Python 3.13.x**: El proyecto requiere específicamente Python 3.13 debido a que la pila de dependencias (TensorFlow 2.21.0, tf-keras 2.21.0, y DeepFace 0.0.100) ha sido verificada y testeada para ser totalmente compatible en esta versión de Python para sistemas Windows.

---

## Configuración de Base de Datos y Entorno

Sigue estos pasos para configurar la base de datos PostgreSQL local y las variables de entorno de Django:

### 1. Crear la Base de Datos en PostgreSQL

Asegúrate de tener instalado PostgreSQL (se recomienda PostgreSQL 15 o superior, probado en PostgreSQL 18) y ejecuta el siguiente comando en tu terminal de PostgreSQL (`psql`) o mediante una herramienta gráfica como pgAdmin:

```sql
CREATE DATABASE emotion_detector_db;
```

O utilizando la terminal de comandos del sistema:

```bash
createdb -U postgres emotion_detector_db
```

### 2. Configurar Variables de Entorno

Copia el archivo de plantilla `.env.example` y nómbralo `.env`:

```bash
cp .env.example .env
```

Abre el archivo `.env` recién creado y ajusta los valores con tus credenciales locales:

```env
# Configuración general de Django
DEBUG=True
SECRET_KEY=tu_clave_secreta_aqui

# Configuración de base de datos PostgreSQL
DB_NAME=emotion_detector_db
DB_USER=postgres
DB_PASSWORD=tu_contraseña_de_postgresql
DB_HOST=127.0.0.1
DB_PORT=5432
```

### 3. Aplicar Migraciones

Una vez configurado el archivo `.env` y activado tu entorno virtual, aplica las migraciones para crear la estructura de tablas de la base de datos ejecutando:

```bash
python manage.py migrate
```

---

## Estructura del Módulo de Detección (Modelos Creados)

* **`DetectionSession`**: Almacena las sesiones iniciadas cuando el stream captura video. Registra el usuario (si existe), inicio, fin y detalles del dispositivo de origen.
* **`EmotionRecord`**: Almacena cada fotograma analizado individualmente de forma relacionada a la sesión con su respectiva emoción detectada (`choices`), nivel de confianza de predicción y coordenadas faciales (`face_bbox` en formato JSON).
