# 👁️ Bogotá Watcher

**Bogotá Watcher** es un sistema inteligente que detecta posibles riesgos en videos relacionados con el entorno urbano, con especial atención en transporte público como TransMilenio.

Detecta automáticamente situaciones como:

- Robos
- Inundaciones
- Trancón (congestión vehicular)
- Aglomeraciones

El sistema usa redes neuronales profundas para analizar videos reales y generar alertas automáticas.

---

**Repositorio:** https://github.com/CamilaG2/Bogota-Watch

---

## 🧠 ¿Qué hace este proyecto?

1. Clasifica escenas peligrosas usando un modelo **MobileNetV2**.
2. Detecta personas con **YOLOv8** para identificar aglomeraciones.
3. Combina ambas tareas para validar coincidencias y generar alertas.
4. Presenta resultados de forma visual usando **Streamlit**.

---

## 🗂️ Estructura del proyecto
```
BogotaWatcher/
├── app.py                  # Streamlit App para procesar un solo vídeo
├── run_pipeline.py         # Script principal que orquesta todo el pipeline
├── requirements.txt        # Dependencias del proyecto
├── README.md               # Este archivo
├── modelo_clasificador.h5  # Este se crea al correr el archivo classify.py
├── yolov8n.pt
│
├── data/                   # Carpeta de datos generados
│   ├── raw/                # Vídeos descargados (.mp4)
│   ├── frames/             # Frames extraídos (por video)
│   ├── detecciones/        # Imágenes con detecciones superpuestas
│   └── clasificador/       # En esta carpeta se encontrará una clasificacion inicial reliazada
│       ├── aglomeracion/
│       ├── inundacion/
│       ├── robo/
│       ├── trancon/
│
└── src/                    # Módulos fuente
    ├── download.py         # Descargar vídeos y lista `urls`
    ├── frames.py           # Extraer frames desde MP4
    ├── detector.py         # Detectar objetos con Ultralytics YOLO
    ├── alerts.py           # Cargar modelo TF y generar alertas
    ├── comparar_alertas.py # Comparar detecciones vs. clasificador
    └── classify.py         # Entrenamiento del clasificador MobileNetV2
```

Se recomienda crear una carpeta llamada data y dentro de ella una llamada raw, esto con el objetivo de guardar lo generado en el archivo download.py y otra llamada frames para guardar lo generado en el archivo frames.py. Adicionalmente, para poder entrenar bien el modelo se recomienda usar la carpeta clasificador ya que en esta se encuentran 4 carpetas que contienen la base para entrenar el modelo.

---

## 🧾 ¿Cómo se puede usar?
| Script          | Herramienta | Ideal para         |
| --------------- | ----------- | ------------------ |
| `app.py`        | Streamlit   | Ejecución local 📍 |

---

## 🛠️ ¿Cómo correr bien el proyecto?

1. Crea y activa un entorno virtual
```bash
# 1. Crear y activar venv
python -m venv venv
# macOS/Linux:
source .venv/bin/activate  
# Windows PowerShell:
venv\Scripts\activate    
# 2. Instalar dependencias  
pip install -r requirements.txt
# 3. Entrenar clasificador
python -m src.classify
# 4. Ejecutar pipeline completo
python run_pipeline.py
# 5. Levantar Streamlit
streamlit run app.py
```
   Si hay un error al activar el entorno, sigue estos pasos:
   ```bash
    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
    venv\Scripts\activate      # En Windows
    pip install -r requirements.txt
    python -m src.classify
    python run_pipeline.py
    streamlit run app.py
   ```
---

## 🔄 Flujo del proyecto

Este es el orden en el que se ejecuta el sistema completo desde cero:

1. **Entrenar el clasificador** con las carpetas del dataset personalizado  
    Por fines prácticos, se debe primero correr el clasificador para que el run_pipeline.py funcione de manera correcta
   Se genera el archivo `modelo_clasificador.h5`.

2. **Descargar videos** desde TikTok  
   Esto guarda los `.mp4` en `data/raw/`.

3. **Extraer frames** de los videos descargados  
   Los frames se guardan en `data/frames/`.

4. **Detectar personas con YOLOv8** en los frames   
   Se guardan imágenes con detecciones en `data/detecciones/`.

5. **Comparar resultados** de ambos modelos  
   Esto imprime coincidencias entre ambos enfoques.

6. **Ejecución** Para tener una visión más clara de lo que realiza el proyecto, al correr el archivo de app.py se abrirá una url en donde se podrán cargar los videos que se quieren analizar.

---

## 📋 Dependencias

| Librería                  | ¿Para qué se usa?                                            |
| --------------------------|------------------------------------------------------------- |
| `yt-dlp>=2023.11.29`      | Para descargar vídeos de TikTok y otras plataformas          |
| `tensorflow>=2.11.0`      | Para cargar y ejecutar el clasificador MobileNetV2 (.h5)     |
| `numpy>=1.21.0`           | Para cálculos numéricos y manipulación de arrays             |
| `Pillow>=8.3.2`           | Para cargar, redimensionar y procesar imágenes               |
| `h5py>=3.1.0`             | Para leer y escribir archivos de modelo en formato HDF5 (.h5)|
| `ultralytics>=8.0.0`      | Para detección de personas con YOLOv8                        |
| `opencv-python>=4.5.3.56` | Para extraer frames de vídeos y operar sobre imágenes        |
| `streamlit>=1.14.0`       | Para desplegar una interfaz web local y probar la aplicación |


---

## 📸 Créditos y dataset
Los videos utilizados fueron descargados desde cuentas públicas de TikTok enfocadas en noticias y seguridad en Bogotá. Las imágenes extraídas se agruparon en clases para crear un dataset personalizado.

---

## 📬 Autoría
Camila Garcia \\
Universidad del Rosario

