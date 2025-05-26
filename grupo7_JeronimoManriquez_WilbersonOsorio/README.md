#  Monitoreo de Ocupación en Transporte Público con Visión por Computador
##  ¿De qué trata este proyecto?

En muchas ciudades, subirse a un bus lleno sin saber cuánto falta para el próximo o si hay espacio es el pan de cada día. Lo que buscamos con este proyecto es simple pero poderoso: usar visión por computador para estimar automáticamente cuán lleno está un bus, con el objetivo de ayudar a mejorar la frecuencia de rutas, informar mejor a los usuarios y evitar aglomeraciones.

Este sistema detecta personas dentro de vehículos de transporte público y clasifica el nivel de ocupación como **bajo**, **medio** o **alto**. Lo hicimos usando deep learning, combinando tareas como detección, segmentación y clasificación. Todo corre en Google Colab y puede integrarse fácilmente a sistemas de monitoreo.

---

##  Arquitectura del sistema

Este proyecto no usa un solo modelo gigante. En cambio, lo pensamos como un sistema compuesto por varias tareas que se conectan entre sí:

1. **Detección de personas**  
   Usamos un modelo YOLOv8 reentrenado con nuestro propio dataset para encontrar personas en imágenes capturadas dentro de buses.

2. **Segmentación**  
   También entrenamos una versión segmentadora de YOLOv8 para obtener la silueta de cada persona. Esto mejora el conteo y nos permite analizar con más precisión la ocupación.

3. **Clasificación del nivel de ocupación**  
   Con base en el número de personas detectadas, clasificamos la escena como:
   - `Low`: poca gente
   - `Medium`: ocupación media
   - `Full`: lleno o casi lleno

Todos estos pasos están integrados en un pipeline que se puede ejecutar con un solo script: `run_pipeline.py`.

---

##  Sobre los datasets

Este proyecto se construyó **desde cero** con imágenes reales capturadas en contextos de transporte público. Todos los datos fueron anotados manualmente usando Roboflow.

###  Dataset de detección
- Tipo: Bounding boxes para la clase `person`
- Formato: YOLOv8
- Fuente: Capturas reales de cámaras en buses y estaciones

### Dataset de segmentación
- Tipo: Máscaras tipo polígono
- Reusamos las imágenes del dataset anterior
- Formato: YOLOv8-seg

### Clasificación por ocupación
Dividimos las imágenes en tres carpetas: `low`, `medium` y `full`, según el número de personas visibles.

---

##  Resultados y métricas

Después de entrenar y evaluar nuestros modelos, obtuvimos resultados bastante sólidos:

###  Detección
- `mAP@0.5`: 91.2%
- `Precisión`: 88.5%
- `Recall`: 93.1%

###  Segmentación
- `mAP@0.5`: 89.7%
- `IoU`: 83.4%

###  Clasificación de ocupación
- Precisión total (Accuracy): **92%**
- Se observaron pequeñas confusiones entre `medium` y `full`, sobre todo en imágenes donde las personas estaban muy juntas o parcialmente ocultas.

---

##  Lo que aprendimos y lo que viene

###  Lecciones aprendidas
- Recolectar y anotar bien los datos es tan importante como entrenar el modelo.
- Combinar varias tareas (detección + segmentación + clasificación) hace el sistema más robusto.
- La modularidad permite entender mejor cada paso y hacer ajustes con mayor facilidad.

###  Posibles mejoras futuras
- Incluir procesamiento de video en tiempo real y técnicas de tracking.
- Adaptar automáticamente los umbrales de clasificación según el tipo de vehículo.
- Crear una app o dashboard que muestre los niveles de ocupación en tiempo real.

---


