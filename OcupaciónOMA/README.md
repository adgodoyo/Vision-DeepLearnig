# Propuesta de Proyecto: Sistema de Detección de Asientos Disponibles mediante Visión Computacional

*Institución:* Universidad del Rosario, Sede Claustro  
*Área del proyecto:* Área del OMA  
*Problema identificado:* Dificultad para encontrar espacios disponibles en zonas comunes (comedores, áreas de socialización)  
*Solución propuesta:* Implementación de un sistema inteligente de monitoreo de asientos  

## Planteamiento del Problema

En la Universidad del Rosario, sede Claustro, los estudiantes enfrentan con frecuencia la dificultad de encontrar asientos disponibles en las áreas comunes destinadas para comer, estudiar o socializar con amigos. Esta situación genera:

1. Pérdida de tiempo al recorrer diferentes espacios físicamente  
2. Frustración al no encontrar lugares disponibles  
3. Congestión en áreas cuando los estudiantes buscan asientos  
4. Dificultad para planificar tiempos de descanso o alimentación  

## Solución Tecnológica Propuesta

Nuestro proyecto implementará un sistema basado en *visión computacional y deep learning* que permitirá:

1. *Optimización del tiempo* estudiantil  
2. *Reducción de congestión* en áreas comunes  
3. *Mejor experiencia* para toda la comunidad universitaria  

## Arquitectura del Programa de Entrenamiento

Para el entrenamiento del modelo de detección de asientos, se siguió una arquitectura de desarrollo basada en pasos secuenciales y modulares, facilitando la integración, evaluación y mejora del sistema. Esta estructura fue implementada en un entorno Jupyter Notebook alojado en Google Colab, utilizando herramientas de deep learning modernas como **YOLOv8**.

### 1. Instalación y configuración del entorno

Se instalaron las librerías necesarias para la detección de objetos, incluyendo `ultralytics` (que proporciona acceso directo a modelos YOLOv8), así como el framework `PyTorch`. Adicionalmente, se integró Google Drive al entorno de trabajo para facilitar el almacenamiento y la lectura de archivos desde la nube.

### 2. Preparación del dataset

El conjunto de datos fue procesado para enlazar correctamente las imágenes capturadas con sus respectivas anotaciones en formato YOLO. Este proceso aseguró que cada imagen tuviera una etiqueta que indicara la ubicación de los asientos, marcados como "silla_ocupada".

### 3. División de datos en entrenamiento y validación

Con el objetivo de evaluar el desempeño del modelo de forma objetiva, el dataset fue dividido en dos subconjuntos:
- **80 %** de los datos fueron destinados al entrenamiento del modelo.  
- **20 %** se reservaron para la validación durante el entrenamiento.  

Los datos se organizaron siguiendo la estructura esperada por YOLOv8, separando imágenes y etiquetas en carpetas específicas (`images/train`, `labels/train`, `images/val`, `labels/val`).

### 4. Configuración del archivo de entrenamiento

Se generó un archivo de configuración en formato `.yaml` donde se definieron los siguientes elementos:
- La ruta al dataset  
- La ubicación de las carpetas de entrenamiento y validación  
- El número de clases (1 clase: "silla_ocupada")  
- Los nombres asociados a cada clase  

Este archivo fue indispensable para que YOLOv8 pudiera interpretar y entrenar correctamente con los datos personalizados del proyecto.

### 5. Entrenamiento del modelo

Se utilizó el modelo base **YOLOv8n** (versión ligera) preentrenado, al cual se le aplicó *fine-tuning* con los datos capturados en el OMA. Se establecieron parámetros clave como:
- Número de épocas: 300  
- Tamaño de imagen: 640×640 píxeles  
- Congelamiento de capas: 10 primeras capas para preservar conocimientos generales  

Esto permitió adaptar el modelo a la tarea específica de detección de asientos ocupados en un entorno universitario real.

---

## Flujo general del programa

1. Instalación de dependencias necesarias  
2. Conexión con Google Drive para carga y almacenamiento  
3. Enlace entre imágenes y sus etiquetas correspondientes  
4. División en conjuntos de entrenamiento y validación  
5. Generación del archivo de configuración `.yaml`  
6. Entrenamiento del modelo YOLOv8 con fine-tuning  


---

## Dataset

El dataset fue creado por los integrantes del grupo, quienes capturaron fotografías en orientación horizontal para asegurar una perspectiva uniforme y adecuada para el análisis.

Para facilitar la recolección de datos en un entorno accesible y con un flujo moderado de personas, se eligió el área del **OMA** dentro de la Universidad del Rosario. Se puso especial énfasis en las **mesas circulares**, ya que representan un reto interesante para la detección y segmentación de asientos debido a su disposición no lineal.

---

## Métricas Empleadas y Discusión de Resultados

Durante el entrenamiento del modelo, se utilizaron las siguientes métricas para evaluar su rendimiento:

- **Precisión (Precision):** mide cuántas de las predicciones realizadas fueron correctas.  
- **Recall (Sensibilidad):** mide cuántas de las verdaderas clases positivas fueron identificadas correctamente.  
- **mAP@0.5 (mean Average Precision):** promedio de precisión considerando una superposición mínima del 50 % entre las predicciones y las anotaciones reales.  

### Resultados Obtenidos

El modelo alcanzó valores satisfactorios en las métricas principales, destacando un **mAP@0.5 superior al 87 %**. Esto indica que el sistema logra identificar con gran precisión las sillas ocupadas en las imágenes analizadas.

### Discusión

Los resultados reflejan un buen desempeño del modelo dentro del entorno específico del OMA. El sistema responde eficazmente a diferentes condiciones lumínicas, disposición de sillas y variabilidad en la ocupación de los espacios. Esto sugiere que su implementación en tiempo real puede ser viable para mejorar la gestión de espacios comunes en la universidad.

---

## Lecciones Aprendidas y Trabajo Futuro

### Lecciones Aprendidas

- **Importancia de un buen dataset:** La calidad y diversidad de las imágenes impactan directamente en la precisión del modelo. La captura desde diferentes ángulos, condiciones de luz y ocupación fue clave para el rendimiento.
- **Simplicidad de integración con YOLOv8:** La librería `ultralytics` facilitó la implementación de detección de objetos de forma rápida y eficiente, incluso para usuarios con conocimientos intermedios.
- **Eficiencia del entrenamiento en Colab:** El uso de Google Colab permitió entrenar el modelo sin requerir recursos locales, aunque con limitaciones en tiempo y GPU.

### Trabajo Futuro
- **Despliegue en tiempo real:** Se plantea adaptar el modelo entrenado para su uso en cámaras en vivo mediante una interfaz web o aplicación móvil.
- **Ampliación del dataset:** Capturar más imágenes en diferentes horarios, días y espacios para mejorar la capacidad de generalización del modelo.


