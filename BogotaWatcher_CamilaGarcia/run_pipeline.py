# run_pipeline.py

from src.download import descargar_videos, urls        # Función y lista de URLs para descargar videos
from src.frames import procesar_videos                # Función para extraer frames de los videos descargados
from src.detector import detectar_en_frames           # Función para detectar objetos en los frames usando YOLO
from src.alerts import cargar_modelo, generar_alertas # Funciones para cargar clasificador y generar alertas CNN
from src.comparar_alertas import comparar_modelos     # Función para comparar resultados entre modelos CNN y YOLO

# Configuración de rutas y parámetros
RAW_DIR             = "data/raw"                      # Carpeta donde se guardan los videos descargados
FRAMES_DIR          = "data/frames"                   # Carpeta donde se extraerán los frames
DETECTIONS_DIR      = "data/detecciones"              # Carpeta donde se guardan las imágenes anotadas por YOLO
MODEL_CLF_PATH      = "modelo_clasificador.h5"       # Ruta del modelo CNN de clasificación
MODEL_YOLO_PATH     = "yolov8n.pt"                   # Ruta del modelo YOLOv8 para detección
UMBRAL_AGLOMERACION = 5                               # Umbral de número de personas para alerta de aglomeración

def run_pipeline():
    # 1) Descarga de vídeos
    print("🔽 Descargando vídeos...")
    descargar_videos(urls, output_dir=RAW_DIR)  # Descarga todos los videos listados en RAW_DIR

    # 2) Extracción de frames
    print("📸 Extrayendo frames...")
    procesar_videos(entrada_dir=RAW_DIR, salida_dir=FRAMES_DIR)  # Extrae frames de cada video en RAW_DIR

    # 3) Detección con YOLO
    print("📦 Ejecutando detección YOLO en frames…")
    detectar_en_frames(
        input_dir=FRAMES_DIR,                  # Carpeta de entrada con los frames
        output_dir=DETECTIONS_DIR,             # Carpeta donde se guardan las detecciones
        modelo_path=MODEL_YOLO_PATH,           # Modelo YOLOv8 a usar
        umbral_aglomeracion=UMBRAL_AGLOMERACION# Umbral para alerta de aglomeración
    )

    # 4) Clasificación y generación de alertas
    print("🚨 Generando alertas de clasificación…")
    modelo_clf = cargar_modelo(MODEL_CLF_PATH)           # Carga el modelo CNN desde disco
    generar_alertas(
        modelo_clf,                                      # Modelo de clasificación cargado
        frames_dir=FRAMES_DIR,                           # Carpeta con los frames a clasificar
        clases=["aglomeracion", "inundacion", "robo", "trancon"]  # Lista de etiquetas posibles
    )

    # 5) Comparación de resultados
    print("🔍 Comparando resultados de modelos…")
    comparar_modelos(
        path_frames=FRAMES_DIR,                         # Carpeta con los frames originales
        modelo_clasificador=MODEL_CLF_PATH,             # Ruta al modelo CNN
        modelo_yolo=MODEL_YOLO_PATH,                    # Ruta al modelo YOLOv8
        umbral_aglomeracion=UMBRAL_AGLOMERACION         # Umbral para comparación de aglomeraciones
    )

    print("✅ Pipeline completado exitosamente.")       # Indicador de fin de ejecución

if __name__ == "__main__":
    run_pipeline()  # Ejecuta el pipeline completo si se ejecuta este archivo directamente
