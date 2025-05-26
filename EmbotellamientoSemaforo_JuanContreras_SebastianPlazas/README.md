# 🚦 Taller Final: Monitoreo Vehicular con Deep Learning

**Grupo:** Juan Sebastián Contreras, Sebastián Plazas Andrade
**Proyecto:** *Monitoreo de Congestión Vehicular en Vías Críticas utilizando Deep Learning*

---

## 🎯 Objetivo General

Diseñar e implementar una solución de visión computacional basada en **Deep Learning** que tenga un impacto social medible.
Nuestro enfoque: **detectar, clasificar, contar y rastrear vehículos** para monitorear la congestión en zonas urbanas críticas, utilizando un modelo **YOLOv11** con *Transfer Learning* mediante *Fine-tuning*.

---

## 🚧 Problemática Abordada

Las ciudades enfrentan serios problemas de tráfico:

* Pérdida de tiempo y productividad.
* Altos niveles de contaminación.
* Infraestructura semafórica poco adaptable.

Este proyecto apoya la **gestión dinámica del tráfico**, mediante el análisis automatizado de video que permite detectar situaciones de congestión en tiempo real a través del conteo, clasificación y seguimiento vehicular.

---

## 🌍 Aplicaciones Similares en el Mundo

| Modelo        | Aporte clave                                                  |
| ------------- | ------------------------------------------------------------- |
| **AlexNet**   | Introducción de ReLU y entrenamiento con GPU                  |
| **GoogLeNet** | Módulos Inception, arquitectura eficiente                     |
| **ResNet**    | Conexiones residuales que resuelven el *vanishing gradient*   |
| **YOLO**      | Detección en una sola pasada, ideal para video en tiempo real |

---

## 🧠 Arquitectura de la Solución

Estructura del proyecto `taller_final/`:

* `README.md`: Documentación principal.
* `segment.ipynb`: Notebook completo con flujo de trabajo, desde detección hasta reporte.
* `yolov11n.pt`: Modelo YOLOv11 entrenado por *fine-tuning*.
* `sort.py`: Algoritmo SORT para seguimiento multiobjeto.
* `analisis_completo.html`: Reporte HTML interactivo con visualizaciones.
* `congestion_por_frame.csv`: Conteo de vehículos y eventos de congestión.
* `tracking_velocidades.csv`: Velocidades y trayectorias por ID.
* `imagenes/`: Resultados visuales (detecciones, mapas de calor, etc.).
* `videos/`: Videos originales.
* `runs/`: Resultados del entrenamiento YOLO.
* `My-First-Project-3/`: Dataset exportado desde Roboflow.
* `__pycache__/`: Archivos temporales generados por Python.

---

## 🧩 Modelo: YOLOv11 + Transfer Learning (Fine-tuning)

### ¿Qué es YOLOv11?

YOLO (*You Only Look Once*) es una red eficiente para **detección de objetos en tiempo real**, con ventajas como:

* Arquitectura optimizada y detección *anchor-free*.
* Rápido entrenamiento y exportabilidad a TensorRT, ONNX, etc.
* Ideal para aplicaciones en producción.

> Usamos **YOLOv11**, última versión con mejoras en precisión, velocidad y eficiencia.
> 🔗 [Más información](https://docs.ultralytics.com/es/models/yolov8/)

### ¿Qué es Fine-tuning?

**Transfer Learning** consiste en reutilizar un modelo preentrenado (ej. en COCO) y adaptarlo a una nueva tarea.

> "Las primeras capas detectan características generales; se ajustan las últimas capas al nuevo contexto."

En nuestro caso, usamos YOLOv11 preentrenado en COCO y lo reentrenamos con imágenes reales tomadas en Bogotá (Puente de Toberín), para detectar vehículos específicos del entorno urbano local.

---

## 🛰️ Seguimiento con SORT

Para rastrear vehículos, usamos **SORT (Simple Online Realtime Tracking)**:

* Basado en filtros de Kalman + IoU.
* Rápido y sin redes neuronales adicionales.
* Asigna un ID persistente a cada vehículo.
* Permite calcular velocidades y trayectorias.

> Aunque no es perfecto, es eficiente y no incrementa el tiempo de procesamiento.

---

## 📚 Librerías Utilizadas

| Librería               | Uso Principal                                       |
| ---------------------- | --------------------------------------------------- |
| `ultralytics`          | Modelo YOLOv11 (entrenamiento y predicción)         |
| `opencv-python`        | Manipulación de video en tiempo real                |
| `roboflow`             | Gestión y exportación del dataset                   |
| `csv`                  | Exportación de datos                                |
| `sort`                 | Seguimiento multiobjeto (tracking ID + coords)      |
| `collections`          | Conteo eficiente por clase (Counter)                |
| `plotly`, `matplotlib` | Visualizaciones interactivas, mapas de calor        |
| `base64`, `PIL`        | Generación de reportes visuales con superposiciones |

---

## 🔄 Flujo del Proyecto

### 1. Captura y Preparación de Datos

* Se toman fotos desde un puente (simulando vista de semáforo):
  ![foto](imagenes/20250522_060645.jpg)
* Se etiquetan en **Roboflow**:
  ![labeling](imagenes/labeling_roboflow2.jpg)
* Se genera y exporta un dataset optimizado:
  ![dataset](imagenes/dataset.png)

### 2. Entrenamiento y Predicción

1. Se importa el modelo `yolov11n.pt`.
2. Se entrena sobre el dataset personalizado.
   ![entrenamiento](runs/detect/entrenamiento_YOLOv11/train_batch0.jpg)
3. Se realiza la inferencia sobre el video urbano real.

### 3. Análisis y Seguimiento

4. Conteo por tipo (`car`, `truck`, `bus`, `motorcycle`) y total.
5. Seguimiento con SORT.
6. Cálculo de velocidades (px/s) por ID.
7. Señales visuales:

   * 🔴 Círculo rojo si hay congestión.
   * Texto: ID + categoría + velocidad.
     ![procesamiento de video](imagenes/ultimo_frame.jpg)

### 4. Reporte Automático

8. Exportación a CSV:

   * `congestion_por_frame.csv`
   * `tracking_velocidades.csv`
9. Generación de **reporte interactivo HTML**:

   * Evolución temporal del tráfico.
   * Velocidad promedio por tipo.
   * Mapa de calor con zonas más transitadas.
   * Superposición de datos sobre el video original.

---

## 🌱 Impacto Social y Replicabilidad

* **Bajo costo:** solo se requiere una cámara fija.
* **Escalable:** aplicable en distintas intersecciones o ciudades.
* **Toma de decisiones basada en datos reales:**

  * Semáforos inteligentes.
  * Análisis de puntos críticos.
  * Priorización de intervenciones según tipo de vehículo.

---

## 🚀 Futuras Mejoras

* Incluir **segmentación semántica** para diferenciar carriles.
* Usar **DeepSORT** o **ByteTrack** para robustez en tracking.
* Añadir **georreferenciación** (OpenStreetMap).
* Integrar lógica adaptativa para **control semafórico**.
* Recolectar datos en distintas **zonas y horarios**.

---