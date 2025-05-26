import cv2                            # Importa OpenCV para manejo de video e imágenes
from pathlib import Path             # Para crear y manipular rutas de manera segura

def extraer_frames_video(video_path, salida_dir="data/frames_app", cada_segundos=1):
    Path(salida_dir).mkdir(parents=True, exist_ok=True)  # Crea la carpeta de salida si no existe
    cap = cv2.VideoCapture(video_path)                   # Abre el video para lectura de frames
    fps = cap.get(cv2.CAP_PROP_FPS)                       # Obtiene la tasa de frames por segundo
    intervalo = int(fps * cada_segundos)                 # Calcula cuántos frames corresponden al intervalo

    i = 0                         # Contador de frames leídos
    guardados = []               # Lista para almacenar rutas de frames guardados
    while True:
        ret, frame = cap.read()  # Lee el siguiente frame
        if not ret:
            break                # Sale del bucle si no hay más frames
        if i % intervalo == 0:   # Si el índice de frame coincide con el intervalo...
            frame_path = f"{salida_dir}/frame_{i:04d}.jpg"  # Nombre formateado del archivo
            cv2.imwrite(frame_path, frame)                # Guarda el frame en disco
            guardados.append(frame_path)                  # Añade la ruta a la lista de guardados
        i += 1                    # Incrementa el contador de frames
    cap.release()                # Libera el recurso de video
    return guardados             # Devuelve la lista de rutas de los frames extraídos
