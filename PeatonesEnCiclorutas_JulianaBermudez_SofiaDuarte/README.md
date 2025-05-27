# Proyecto Final
Por: Juliana Bermúdez y Sofía Duarte

## Resumen del problema y su impacto social

En Bogotá hay ciclorutas amplias en varias vías principales, espacios donde los biciusuarios pueden transitar con seguridad. De esta manera, se incentiva el transporte sostenible, ya que no genera emisiones de gases contaminantes y se está combatiendo el sedentarismo cada vez más generalizado y nocivo para la salud de los ciudadanos. Sin embargo, dentro de estas vías destinadas para el transito de bicicletas tienden a actuar factores externos que dificultan la movilización segura; siendo el más común el tránsito de peatones. En varias ocasiones las personas salen a trotar, caminar o simplemente se sitúan sobre estas secciones del andén poniendo en peligro sus vidas y las de los biciusuarios. Por tanto, proponemos una solución que sea capaz de identificar las ciclorutas y cuando se encuentren peatones sobre ellas para alertar que hay peligro. De esta manera, estamoso velando por la seguridad de los peatones y biciusuarios, los cuales tienen la mayor prelación (respectivamente) en el código nacional de tránsito.  


## Descripción de la arquitectura y justificación de decisiones

Dada la tarea, decidimos implementar en primer lugar un modelo de segmentación que fuera capaz de demarcar las  ciclorutas. Posteriormente, un modelo de detección que identificara personas. Finalmente, entre los parámetros del modelo se encuentran las coordenadas en las que se identificaron los objetos y sus etiquetas correspondientes. Con esto, hacemos las "cajas" en las que están las personas y las ciclovías para superponerlas y ver si hay intersección. De esa manera, sabemos si hay personas en la ciclovía.

### Modelo de segmentación

Para esta tarea, decidimos utilizar yolo11n-seg.pt de la librería ultralytics. Este modelo de segmentación, se entrena sobre el conjunto de datos COCO y tiene las siguientes especificaciones:

| Modelo        | tamaño (píxeles) | mAP<sub>box</sub> 50-95 | mAP<sub>mask</sub> 50-95 | Velocidad CPU ONNX (ms) | Velocidad T<sub>4</sub> TensorRT10 (ms) | parámetros (M) | FLOPs (B) |
|---------------|------------------|--------------------------|---------------------------|--------------------------|------------------------------------------|----------------|-----------|
| YOLOv11n-seg | 640              | 38.9                     | 32.0                      | 65.9 ± 1.1              | 1.8 ± 0.0                                | 2.9            | 10.4      |

Al momento de probarlo sobre una de nuestras imágenes vimos que no segmentaba lo que deseábamos, por lo que decidimos hacer fine-tuning. Para ello, fue necesario hacer un preprocesamiento de las imágenes utilizando la aplicación web CVAT. Esta permite tomar las imagenes y manualmente demarcar con los labels de interés, para posteriormente descargarlas en el formato necesario para el modelo que se vaya a emplear. En nuestro caso, solo utilizamos labels para delimitar las ciclovias y descargamos las imagenes de tal manera que tuviesen labels en formato txt. De hecho, los datos se tuvieron que acomodar de la siguiente manera:
```
dataset/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
```
donde en images set tienen las imágenes como fueron tomados y en labels lo obtenido usando CVAT.

Después de tener las imágenes, se creo un archivo `/src/data.yaml` en el que se especificaron los paths de las imagenes tanto para entrenamiento como para validación. Además, se especificaron las clases que ibamos a utilizar y como estaban codificadas. Este archivo fue importante, ya que esto se ingresó al modelo para el entrenamiento e inmediatamente supo cuales imagenes debía tomar y cuantas clases debía tomar para reentrenar la última capa (la de clasificación). De hecho, se probó congelar hasta las últimas 5 capas de ACÁ TOCA PONER EL NÚMERO, pero los resultados no fueron más satisfactorios que al reentrenar exclusivamente la última capa. Por tanto, decidimos dejar el entrenamiento con la última capa por temas de eficiencia computacional.

### Modelo de detección

Para esta tarea, decidimos utilizar yolo11n.pt de ultralytics el cual fue entrenado y probado sobre el conjunto de datos COCO. Las razones fueron las siguientes:

- Extracción de características mejorada por tener una arquitectura de columan vertebral y cuello.

- Optimizado para eficiencia y velocidad, al igual que mayor precisión co nmenos parámetros en comparación con YOLOv8m. De hecho, tiene 22% menos parámetros sobre el conjunto de datos de COCO sin comprometer precisión.

- Se puede implementar en distintos entornos como dispositivos periféricos, plataformas en la nube y sistemas compatibles con GPU NVIDIA.

Al hacer pruebas,vimos que es demasiado bueno identificando personas e incluso bicicletas. Por tanto, no vimos necesario hacer fine-tuning.

### Modelo final

Dado que los modelos dan las esquinas de los bloques en los que identificó objetos, tomamos los bloques de la ciclovía (dados por el modelo de segmentación) y los de personas (dados por el modelo de detección) y buscamos intersecciones. Dado que queremos poderlo implementar en otro documento, lo creamos como una función llamada `check_person_in_bike_lane` que toma los siguientes parámetros:

- img_path: Ruta a la imagen de entrada.

- bike_lane_model: Modelo de segmentación entrenado para detectar ciclovías.

- person_model: Modelo de detección entrenado para detectar personas.

- bike_lane_class_id: ID de clase de ciclovía en la salida del modelo de segmentación (en nuestro caso es 0).

- person_class_id: ID de clase de persona en el modelo de detección (en nuestro caso es 0).

- bicycle_class_id: ID de clase de bicicleta en el modelo de detección (en nuestro caso es 1).

- conf: Umbral de confianza para las predicciones.


A partir de esto, hacemos uso de varias herramients de cv2 para poder crear los bloques e intersectarlos sobre las imagenes ya que necesitabamos hacer que las máscaras de todas las ciclovías se volvieran binarias con una resolución que coincida con la imagen original. Además, teniamos que extraer las cajas delimitadoras de los objetos detectados como personas según el id que el modelo tenía para personas, y lo mismo para bicicletas. Después, para descartar las personas que montan en bicicleta, se buscan intersecciones entre cajas de personas y bicicletas. Si la métrica de IOU es mayor a 0.2, se asume que la persona está montando bicicleta y se descarta. Después, para cada persona detectada, tomamos la sección inferior de la caja, que asumimos que en la mayoría de los casos detectará lo spies de la persona. Luego recortabamos dicha area sobre la mascara de ciclovía y si había intersección de pixeles, se decide que la persona está caminando sobre la ciclovía.

Finalmente, se utilizó matplotlib para mostrar la segmentación de la ciclovía en rojo y las personas detectadas con una caja verde sobre la imagen original.


## Detalle de los datasets (propios y externos)

Creamos nuestro propio dataset tomando fotos de diversas ciclorutas en el norte de Bogotá. Además, hicimos el proceso descrito previamente con CVAT para obtener los labels en formato txt requeridos para el entrenamiento del modelo de segmentación.

## Métricas empleadas y discusión de resultados

Utilizamos las métricas propias del modelo que son mAP@0.5:0.95 y mAP@0.5. La primera, mAP@0.5:0.95 (mean Average Precision), es una medida clave para evaluar modelos de detección o segmentación como YOLO. Calcula la precisión promedio (AP) de cada clase considerando diferentes umbrales de IoU (Intersection over Union), desde 0.5 hasta 0.95 en pasos de 0.05. En esencia, el modelo debe predecir correctamente no solo qué objetos hay, sino también localizarlos con precisión. Mientras que mAP@0.5 solo requiere una superposición del 50% entre la predicción y el objeto real, mAP@0.5:0.95 evalúa el desempeño de manera más rigurosa, exigiendo desde una coincidencia moderada hasta casi exacta. Esta métrica es el estándar en conjuntos de datos como COCO, ya que refleja de manera más realista la capacidad del modelo para detectar y localizar objetos con precisión.

A pesar de intentar hacer fine-tuning con nuestro dataset en varias capas, obtuvimos métricas de 0.36 para mAP@0.5 y de 0.32 para mAP@0.5:0.95. Estos valores no son muy buenos. Creemos que se puede deber a que contamos con un dataset pequeño para los estándares de este tipo de modelos, pero hacer aumento de datos con transformaciones para genrar un crecimiento exponencial en el tamaño no era viable ya que se tenían que demarcar las clases manualmente en CVAT para garantizar que los labels estuvieran bien. Adicionalmente, creemos que uno de los grandes problemas es que hay ciclorutas donde la demarcación no era tan clara o difería en temas de color, entonces esto combinado con el dataset reducido no permitió que el modelo verdaderamente las identificara.

Sin embargo, en la práctica, los resultados de segmentación funcionan lo suficientemente bien para cumplir con el objetivo del proyecto.

En cuanto a la combinación del modelo de segmentación y detección, los resultados fueron buenos. En general, el modelo logra detectar correctamente a los peatones en la cicloruta, descartando a los ciclistas. Además, la elección del tamaño de la caja restringida solo para los pies funcionó correctamente, y el modelo pudo diferenciar entre peatones con los pies sobre la ciclovía y aquellos que caminaban justo al lado.

Sin embargo, hay espacio para mejoras. En primer lugar, tuvimos que utilizar un nivel de confianza de 0.1 para la segmentación de ciclorutas, dado que no obteníamos resultados buenos con niveles mayores. Además, el modelo no diferencia entre peatones y personas en patineta, dado que el modelo de detección no está entrenado para detectar patinetas. Sin embargo, en términos generales el modelo funciona correctamente.


## Lecciones aprendidas y trabajo futuro

- Habría sido útil encontrar una forma más eficiente de generar los labels con la precisión deseada. Como nosotras queríamos que los labels fueran exactamente de la ciclovía los generamos manualmente haciendo difícil la escalabilidad y por tanto limitando el conjunto de datos que pudimos utilizar.

- En cuanto a las ciclovías, la demarcación irregular de las mismas pudo haber hecho difícil que el modelo las identificara como algo distinto a una calle normal.

- Pudo haber sido más exitoso emplear modelos más robustos de yolo11, pero dada la capacidad computacional limitada con la que contabamos no se hizo. Estos pudieron haber aportado resultados más acorde a lo esperado y con métricas positivas.

- Al momento de unir los modelos pensabamos que teníamos que hacer otro modelo de deep learning, pero nos dimos cuenta que no siempre hay que hacer algo tan complejo para tareas que pueden salir de manera más rápida y con mucho menor costo computacional. A veces nos acostumbramos a hacer todo con los modelos más robustos, pero una intersección resultó ser suficiente para nuestros propósitos.

## Referencias

[1] CVAT Built-in Data Annotation Instructions
CVAT Team. "Built-in data annotation instructions in CVAT." [Online]. Available: https://www.cvat.ai/resources/blog/built-in-data-annotation-instructions-in-cvat. [Accessed: May 25, 2025].

[2] Ultralytics. "Ultralytics YOLOv8." [Online]. Available: https://docs.ultralytics.com/models/yolo11/#citations-and-acknowledgements. [Accessed: May 25, 2025].

[3]Ultralytics. "Instance Segmentation - Ultralytics YOLO Docs." [Online]. Available: https://docs.ultralytics.com/tasks/segment/. [Accessed: May 25, 2025].
