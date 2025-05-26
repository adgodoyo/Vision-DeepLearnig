from ultralytics import YOLO            # Importa la clase YOLO para realizar detección de objetos
import cv2                              # OpenCV para leer y guardar imágenes
import os                               # Módulo para interactuar con el sistema de archivos
from pathlib import Path               # Para manejar rutas de forma segura y crear directorios

def detectar_en_frames(
    input_dir="data/frames",            # Directorio de entrada con subcarpetas de frames por video
    output_dir="data/detecciones",      # Directorio donde se guardarán las imágenes resultantes
    modelo_path="yolov8n.pt",           # Ruta al archivo del modelo YOLOv8
    umbral_aglomeracion=5               # Número mínimo de personas para disparar alerta
):
    model = YOLO(modelo_path)           # Carga el modelo YOLOv8 especificado
    Path(output_dir).mkdir(             # Crea el directorio de salida si no existe
        parents=True,
        exist_ok=True
    )

    # Recorre cada carpeta dentro de input_dir (cada carpeta representa un video)
    for carpeta in os.listdir(input_dir):
        carpeta_path = Path(input_dir) / carpeta
        if not carpeta_path.is_dir():
            continue                     # Omite entradas que no sean carpetas

        # Prepara la carpeta de salida específica para este video
        out_folder = Path(output_dir) / carpeta
        out_folder.mkdir(                # Crea la carpeta de salida para este video
            parents=True,
            exist_ok=True
        )

        # Itera sobre cada archivo dentro de la carpeta del video
        for img_name in os.listdir(carpeta_path):
            if img_name.endswith(".jpg"):  # Procesa solo archivos con extensión .jpg
                img_path = str(carpeta_path / img_name)  # Ruta completa de la imagen
                results = model(img_path)               # Ejecuta la detección YOLOv8
                result = results[0]                     # Toma el primer (y único) resultado
                img_result = result.plot()              # Dibuja cajas y etiquetas sobre la imagen

                # Cuenta cuántas detecciones corresponden a la clase “persona” (class_id 0)
                conteo_personas = sum(
                    1 for r in result.boxes.cls if int(r) == 0
                )

                out_path = str(out_folder / img_name)   # Ruta donde se guardará la imagen resultante
                cv2.imwrite(out_path, img_result)       # Guarda la imagen anotada

                # Imprime el número de personas detectadas en este frame
                print(f"✅ {img_name} → {conteo_personas} persona(s) detectadas")

                # Si el conteo supera el umbral, emite una alerta de aglomeración
                if conteo_personas >= umbral_aglomeracion:
                    print(
                        f"🚨 ALERTA DE AGLOMERACIÓN: "
                        f"{conteo_personas} personas en {img_name} (video: {carpeta})"
                    )
