import tensorflow as tf  # Importa TensorFlow para cargar el modelo de clasificación
from tensorflow.keras.preprocessing import image  # Utilidades de Keras para cargar y procesar imágenes
import numpy as np  # Para operaciones numéricas y manejo de arrays
from ultralytics import YOLO  # Clase YOLO para detección de objetos
import os  # Para interactuar con el sistema de archivos
from pathlib import Path  # Para manipular rutas de forma más segura

def comparar_modelos(path_frames="data/frames", modelo_clasificador="modelo_clasificador.h5", modelo_yolo="yolov8n.pt", umbral_aglomeracion=5):
    # Carga el modelo de clasificación desde el archivo .h5
    model_clasificador = tf.keras.models.load_model(modelo_clasificador)
    # Inicializa el modelo YOLO para detección de objetos
    model_yolo = YOLO(modelo_yolo)

    # Lista de nombres de clases en el mismo orden que el modelo de clasificación
    clases = ["aglomeracion", "inundacion", "robo", "trancon"]

    # Lista donde se almacenarán los resultados de cada frame procesado
    resultados = []

    # Recorre cada subcarpeta dentro de path_frames (cada carpeta representa un video)
    for carpeta in os.listdir(path_frames):
        carpeta_path = Path(path_frames) / carpeta
        if not carpeta_path.is_dir():
            continue  # Si no es una carpeta, la ignora

        # Recorre cada archivo dentro de la carpeta del video
        for frame in os.listdir(carpeta_path):
            if not frame.endswith(".jpg"):
                continue  # Solo procesa archivos .jpg

            frame_path = str(carpeta_path / frame)  # Construye la ruta completa al frame

            # → Clasificador: carga y preprocesa la imagen para el modelo CNN
            img = image.load_img(frame_path, target_size=(224, 224))  # Carga y redimensiona
            x = image.img_to_array(img)  # Convierte a array NumPy
            x = np.expand_dims(x, axis=0) / 255.0  # Añade dimensión de batch y normaliza
            pred = model_clasificador.predict(x)  # Obtiene vector de probabilidades
            clase_idx = np.argmax(pred)  # Índice de la clase con mayor probabilidad
            pred_clasificador = clases[clase_idx]  # Nombre de la clase predicha

            # → Detección YOLO: ejecuta el detector sobre el frame
            result = model_yolo(frame_path)[0]  # Primer conjunto de resultados
            # Cuenta cuántas detecciones corresponden a la clase “persona” (cls == 0 en COCO)
            conteo_personas = sum(1 for r in result.boxes.cls if int(r) == 0)
            # Determina si supera el umbral para considerar aglomeración
            alerta_yolo = conteo_personas >= umbral_aglomeracion

            # Comparación: ambas alertas (clasificador y YOLO) deben indicar aglomeración
            coincide = alerta_yolo and pred_clasificador == "aglomeracion"

            # Guarda en la lista un diccionario con la información del frame
            resultados.append({
                "video": carpeta,
                "frame": frame,
                "clasificador": pred_clasificador,
                "personas_detectadas": conteo_personas,
                "yolo_alerta": alerta_yolo,
                "coincide": coincide
            })

            # Imprime un resumen por consola
            print(f"{frame} | Clasificador: {pred_clasificador} | Personas: {conteo_personas} | Coincide: {coincide}")

    # Devuelve la lista completa de resultados
    return resultados
