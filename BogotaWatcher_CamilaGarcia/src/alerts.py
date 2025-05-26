import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from pathlib import Path

def cargar_modelo(modelo_path="modelo_clasificador.h5"):
    """
    Carga un modelo de Keras desde el archivo especificado.
    
    Parámetro:
    - modelo_path (str): Ruta al archivo .h5 del modelo.

    Retorna:
    - modelo (tf.keras.Model): Modelo cargado listo para inferencia.
    """
    return tf.keras.models.load_model(modelo_path)


def predecir_frame(modelo, frame_path, img_size=224):
    """
    Realiza la predicción de clase para una sola imagen (frame).

    Parámetros:
    - modelo (tf.keras.Model): Modelo cargado para la predicción.
    - frame_path (str): Ruta al archivo de imagen.
    - img_size (int): Tamaño al que se redimensiona la imagen (img_size x img_size).

    Flujo interno:
    1. Carga la imagen y la redimensiona.
    2. Convierte la imagen a un array NumPy de forma (1, img_size, img_size, 3).
    3. Normaliza los valores de píxel a [0, 1].
    4. Llama a modelo.predict() para obtener la probabilidad por clase.
    5. Devuelve el índice de la clase más probable y el vector de probabilidades.

    Retorna:
    - clase_predicha (int): Índice de la clase con mayor probabilidad.
    - pred (np.ndarray): Array de probabilidades de cada clase.
    """
    # 1. Carga y redimensiona la imagen
    img = image.load_img(frame_path, target_size=(img_size, img_size))
    # 2. Convierte a array (height, width, channels)
    x = image.img_to_array(img)
    # 3. Crea un batch de tamaño 1 y normaliza
    x = np.expand_dims(x, axis=0) / 255.0
    # 4. Inferencia: obtiene probabilidades
    pred = modelo.predict(x)
    # 5. Selecciona la clase de máxima probabilidad
    clase_predicha = np.argmax(pred)
    return clase_predicha, pred


def generar_alertas(modelo, frames_dir="data/frames", clases=None):
    """
    Recorre los directorios de frames y genera alertas
    cuando se detectan situaciones críticas.

    Parámetros:
    - modelo (tf.keras.Model): Modelo para realizar predicciones.
    - frames_dir (str): Directorio raíz que contiene subcarpetas por video.
    - clases (list of str): Lista de nombres de clases en el orden de salida del modelo.

    Flujo interno:
    1. Si no se proporciona, asigna nombres por defecto: 
       ["aglomeracion", "inundacion", "robo", "trancon"].
    2. Recorre cada subcarpeta dentro de frames_dir (un video).
    3. Dentro de cada subcarpeta, procesa solo archivos .jpg.
    4. Para cada frame:
       a. Llama a predecir_frame() para obtener la clase.
       b. Mapea el índice a su nombre de clase.
       c. Si la clase es una de las críticas ("robo", "inundacion", "trancon"),
          imprime un mensaje de alerta con el nombre del frame y del video.
    """
    # 1. Definir nombres de clases si no vienen dados
    if clases is None:
        clases = ["aglomeracion", "inundacion", "robo", "trancon"]

    # 2. Iterar sobre cada carpeta (video) en el directorio de frames
    for carpeta in os.listdir(frames_dir):
        carpeta_path = os.path.join(frames_dir, carpeta)
        # Ignorar archivos que no sean carpetas
        if not os.path.isdir(carpeta_path):
            continue

        # 3. Iterar sobre cada imagen JPG en la carpeta
        for frame_file in os.listdir(carpeta_path):
            if frame_file.lower().endswith(".jpg"):
                frame_path = os.path.join(carpeta_path, frame_file)
                # 4a. Predecir la clase del frame
                clase_idx, _ = predecir_frame(modelo, frame_path)
                # 4b. Obtener nombre de clase
                clase = clases[clase_idx]

                # 4c. Si es una situación crítica, emitir alerta
                if clase in ["robo", "inundacion", "trancon"]:
                    print(
                        f"🚨 ALERTA: Se detectó '{clase.upper()}' "
                        f"en {frame_file} del video '{carpeta}'"
                    )
