# Proyecto Final: Identificación automatizada de rampas urbanas para movilidad inclusiva

**Por: Valentina Hernández Quintana y Laura Alejandra Rincón Castaño**

---

## 🧩 Resumen del problema y su impacto social

En muchas ciudades, las rampas peatonales son un elemento clave para garantizar la movilidad inclusiva, especialmente para personas con discapacidad, adultos mayores o personas con coches de bebé. Sin embargo, la ausencia de registros actualizados y sistemas de monitoreo dificulta su adecuada planificación, mantenimiento y uso.

Este proyecto propone una solución basada en visión por computador para **identificar automáticamente rampas urbanas en imágenes**, con el fin de mapearlas y generar un sistema de apoyo para decisiones de infraestructura inclusiva.

---

## 🛠️ Descripción de la arquitectura y decisiones de diseño

La solución está compuesta por un pipeline que combina detección y segmentación de rampas, georreferenciación de los resultados y publicación en la nube mediante GitHub Pages. A continuación, se detallan los componentes clave:

### 🔍 Modelo de detección

Se utilizó **YOLOv11n** para identificar la presencia de rampas en imágenes urbanas. Aunque el modelo está preentrenado en COCO, se realizó fine-tuning con un dataset propio utilizando Roboflow y configuraciones personalizadas.

### ✂️ Modelo de segmentación

Se implementó **YOLOv11n-seg** para obtener una máscara precisa de la rampa en cada imagen. También se realizó entrenamiento sobre datos anotados manualmente para mejorar el desempeño en condiciones locales.

### 🧭 Georreferenciación

A partir de un archivo `.csv` con coordenadas de latitud y longitud por imagen, se tradujeron las posiciones de detección (en píxeles) a coordenadas geográficas, estimando así la ubicación real de cada rampa detectada.

### ☁️ Publicación web

El resultado se guarda en un archivo `.csv` y se sube automáticamente a un repositorio de GitHub, desde donde se puede visualizar de manera interactiva a través de **GitHub Pages**.

---

## 🗃️ Estructura del pipeline

1. Instalación de dependencias (`ultralytics`, `roboflow`, `opencv`, `matplotlib`, etc.)
2. Descarga de datasets desde Roboflow.
3. Entrenamiento de modelos YOLO (detección y segmentación).
4. Inferencia sobre imágenes de prueba.
5. Conversión de coordenadas píxel → geográficas.
6. Visualización con `matplotlib`.
7. Exportación a CSV y subida a GitHub.
8. Visualización en: [Mapa de rampas](https://laurar287.github.io/Mapa-rampas/)

---

## 🧪 Datasets utilizados

* **Propios**: Imágenes recolectadas en entornos urbanos reales.
* **Roboflow**: Uso de la plataforma para anotación, gestión y descarga de los datasets en formato compatible con YOLOv11.

Los datasets fueron preparados siguiendo la estructura requerida por Ultralytics:

```
dataset/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
```

---

## 📈 Métricas y resultados

* **Detección**: Se alcanzó una precisión satisfactoria con `mAP@0.5 ≈ 0.47` y `mAP@0.5:0.95 ≈ 0.34` en el dataset personalizado.
* **Segmentación**: Se lograron resultados útiles para superponer máscaras que ayudan a validar y visualizar la posición precisa de las rampas.
* **Georreferenciación**: La conversión de píxeles a coordenadas funcionó con buena aproximación para visualización en mapas urbanos.

---

## 🧠 Lecciones aprendidas y mejoras futuras

* La combinación de modelos de segmentación y detección puede fortalecer la precisión del sistema.
* Es fundamental disponer de datasets de mayor tamaño y variedad para mejorar el rendimiento de los modelos.
* El proceso de anotación manual en plataformas como CVAT o Roboflow requiere tiempo, pero mejora la calidad del entrenamiento.
* Se podrían integrar técnicas de aumento de datos y mapas base tipo Leaflet o Folium para mejorar la visualización geográfica.
* Implementar una app móvil que consuma este servicio permitiría escalar el impacto del proyecto.

---

## 📦 Requisitos (requirements.txt)

```txt
ultralytics==8.0.20
roboflow==1.1.17
matplotlib==3.7.1
PyGithub==1.59.0
opencv-python-headless==4.7.0.72
pandas==1.5.3
numpy==1.22.4
requests==2.28.2
```

---

## 📚 Referencias

1. Ultralytics YOLO Docs: [https://docs.ultralytics.com/](https://docs.ultralytics.com/)
2. Roboflow Docs: [https://docs.roboflow.com/](https://docs.roboflow.com/)
3. GitHub PyGithub Docs: [https://pygithub.readthedocs.io/](https://pygithub.readthedocs.io/)
