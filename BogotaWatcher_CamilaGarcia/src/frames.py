import cv2                            # OpenCV: manejo de video e imágenes
import os                             # Interacción con el sistema de archivos
from pathlib import Path             # Manejo seguro de rutas de archivo

def extraer_frames(video_path, salida_dir, intervalo_segundos=1):
    video_name = Path(video_path).stem               # Nombre base del archivo de video (sin extensión)
    salida_path = Path(salida_dir) / video_name      # Carpeta de salida por video
    salida_path.mkdir(parents=True, exist_ok=True)   # Crea la carpeta si no existe

    cap = cv2.VideoCapture(str(video_path))          # Abre el video para lectura de frames
    fps = cap.get(cv2.CAP_PROP_FPS)                  # Obtiene frames por segundo del video
    intervalo_frames = int(fps * intervalo_segundos) # Convierte segundos a frames
    i, guardados = 0, 0                              # Contadores: i para frames totales, guardados para extras

    while True:
        ret, frame = cap.read()                      # Lee el siguiente frame
        if not ret:
            break                                    # Sale al llegar al final del video
        if i % intervalo_frames == 0:                # Si corresponde al intervalo deseado...
            out_path = salida_path / f"frame_{guardados:03}.jpg"  # Nombre de archivo con índice
            cv2.imwrite(str(out_path), frame)        # Guarda el frame como imagen JPG
            guardados += 1                           # Incrementa contador de guardados
        i += 1                                       # Incrementa contador total de frames
    cap.release()                                    # Libera el recurso de video
    print(f"🖼️ {guardados} frames guardados para {video_name}")  # Reporte de número de frames extraídos

def procesar_videos(entrada_dir="data/raw", salida_dir="data/frames"):
    Path(salida_dir).mkdir(parents=True, exist_ok=True)  # Asegura la carpeta raíz de frames
    for archivo in os.listdir(entrada_dir):              # Itera por archivos en el directorio de entrada
        if archivo.endswith(".mp4"):                     # Filtra solo videos MP4
            ruta = os.path.join(entrada_dir, archivo)    # Construye la ruta completa al video
            extraer_frames(ruta, salida_dir)             # Llama a la función de extracción de frames
