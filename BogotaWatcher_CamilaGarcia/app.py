import streamlit as st                  # Importa Streamlit para construir la interfaz web
from PIL import Image                   # PIL para manejar imágenes (no usada directamente aquí)
from pathlib import Path                # Para manipular rutas de archivo de forma segura
import os                               # Para operaciones de sistema de archivos
from src.utils import extraer_frames_video  # Función propia para extraer frames de un video
from src.alerts import cargar_modelo, predecir_frame  # Funciones propias para carga y predicción del clasificador
from ultralytics import YOLO            # Importa YOLO para detección de objetos

# Configuración general de la página Streamlit
st.set_page_config(
    page_title="Bogotá Watch",         # Título de la pestaña del navegador
    layout="wide"                      # Diseño ancho para aprovechar todo el espacio
)
st.title("🚨 Bogotá watch – Detección de riesgos en video")  # Título principal en la app

# Sidebar: carga de video
st.sidebar.markdown("📤 Sube un video .mp4 para analizar")
video_file = st.sidebar.file_uploader(
    "Selecciona un video",             # Etiqueta del uploader
    type=["mp4"]                       # Solo aceptar archivos MP4
)

# Si el usuario ha subido un archivo...
if video_file is not None:
    video_name = Path(video_file.name).stem  # Nombre base del video sin extensión
    temp_video_dir = Path("temp")            # Carpeta temporal para almacenar el video
    temp_video_dir.mkdir(exist_ok=True)      # Crea la carpeta si no existe
    video_path = temp_video_dir / video_file.name  # Ruta completa al archivo temporal

    # Guarda el archivo subido en disco
    with open(video_path, "wb") as f:
        f.write(video_file.read())
    st.video(str(video_path))                # Muestra el video en la app
    st.success("✅ Video cargado. Extrae frames y analiza.")  # Mensaje de éxito

    # Botón para iniciar el procesamiento
    if st.button("🔍 Procesar video"):
        st.info("📸 Extrayendo frames y detectando objetos...")  # Mensaje informativo

        # Extraer frames del video usando la función utilitaria
        frame_dir = f"data/frames_app/{video_name}"  
        os.makedirs(frame_dir, exist_ok=True)    # Crear directorio de frames
        frames = extraer_frames_video(str(video_path), frame_dir)

        # Cargar los modelos de detección y clasificación
        modelo_yolo = YOLO("yolov8n.pt")        # Modelo YOLOv8 para detección de personas
        modelo_clasificador = cargar_modelo("modelo_clasificador.h5")  # CNN para clasificación de riesgo
        clases = ["aglomeracion", "inundacion", "robo", "trancon"]  # Etiquetas del clasificador

        alertas_detectadas = 0  # Contador de alertas emitidas

        # Procesar cada frame extraído
        for frame_path in frames:
            st.markdown(f"### Frame: {Path(frame_path).name}")  # Encabezado por frame

            # Clasificación del frame
            clase_idx, _ = predecir_frame(modelo_clasificador, frame_path)
            clase = clases[clase_idx]
            st.write(f"🧠 Clasificación: **{clase.upper()}**")  # Muestra la clase predicha

            # Detección de objetos con YOLO
            result = modelo_yolo(frame_path)[0]  # Ejecuta el modelo en la imagen
            result_img = result.plot()           # Genera imagen con cajas dibujadas
            st.image(result_img, caption="🔎 Detección YOLO")  # Muestra la imagen anotada

            # Si la clasificación indica un riesgo crítico, muestra alerta
            if clase in ["robo", "inundacion", "trancon"]:
                st.error(f"🚨 ALERTA: {clase.upper()} detectado")
                alertas_detectadas += 1

        # Mensaje final según si se detectaron alertas
        if alertas_detectadas == 0:
            st.success("✅ No se detectó ningún peligro en este video.")
