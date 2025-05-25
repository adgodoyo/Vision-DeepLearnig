# Sistema de Visión Computacional para Control de Inventario Doméstico

### Alejandro Vega - Jose Mejia 

---

## 1. Resumen del problema y su impacto social

Cada año se desperdician toneladas de alimentos en hogares por falta de control sobre lo que se tiene en la nevera o la despensa, muchos productos expiran sin ser consumidos, generando un impacto económico y ambiental negativo.

Este proyecto busca mitigar ese problema mediante un sistema de visión por computador que:

- Detecta automáticamente productos en la nevera mediante cámara.
- Permite llevar un control visual y actualizado del inventario doméstico.

### Enfoque social:
Este sistema busca reducir el desperdicio de alimentos, ahorrar dinero en compras innecesarias y fomentar el consumo responsable en el hogar, puede aplicarse en hogares, residencias estudiantiles, comedores comunitarios o incluso en tiendas.

---

## 2. Descripción de la arquitectura y desarrollo por fases

El sistema fue desarrollado en dos componentes principales, integrados en un pipeline sencillo:

### Fase 1: Detección de productos con YOLOv11

- **Modelo:** YOLOv11 entrenado con un dataset propio compuesto por:
  - Imágenes tomadas en supermercados colombianos.
  - Fotografías de productos reales en la nevera de los autores.
- **Entrenamiento:** 80 épocas, análisis de validación con mAP.
- **Resultado:** El modelo detecta correctamente productos como mermelada, atún, mostaza, entre otros.
- **Aplicación:** Se generan bounding boxes para cada producto identificado en el frame.

### Fase 2: Detección en tiempo real por cámara

- **Tecnología:** Uso de la biblioteca OpenCV para capturar video desde webcam.
- **Funcionalidad:** El sistema permite ver en vivo qué productos están presentes frente a la cámara.
- **Visualización:** Se anotan en pantalla los productos reconocidos en cada frame, lo cual permite construir el inventario visual de forma inmediata.

---

## 3. Detalle de los datasets

| Dataset                   | Tipo   | Fuente                   | Descripción                                                  |
|---------------------------|--------|--------------------------|--------------------------------------------------------------|
| Productos de supermercado | Propio | Fotografías del grupo    | Imágenes de estantes reales en tiendas y supermercados       |
| Productos en la nevera    | Propio | Fotografías del grupo    | Imágenes con fondo doméstico de productos reales             |

El dataset fue completamente recolectado por los autores, etiquetado en formato YOLO con clases como 'leche', 'atún', 'mayonesa', 'queso parmesano', entre otros.

---

## Ejecución del pipeline en local

### Estructura esperada:

```
  /ReconocimientoAlimentos_AlejandroVega_JoseMejia/
  ├── run_pipeline.py
  ├── requirements.txt
  ├── src/
  │   └── runs/detect/train/weights/best.pt
```

### Pasos:

1. Instala dependencias:
```bash
pip install -r requirements.txt
```

2. Ejecuta el pipeline:
```bash
python run_pipeline.py
```

3. Usa la cámara web para mostrar productos al sistema. Pulsa 'q' para salir.

---

## 4. Lecciones aprendidas y trabajo futuro

### Lecciones aprendidas:
- La calidad del dataset propio es determinante para la precisión del modelo.
- El modelo YOLOv11 es rápido y preciso, incluso con pocos datos bien anotados.
- La detección en tiempo real requiere buena iluminación para funcionar correctamente.

### Trabajo futuro:
- Usar OCR para leer fechas de vencimiento en los productos.
- Crear un sistema de alerta para productos vencidos o faltantes.
- Llevar un historial de entradas y salidas por tracking o matching visual.

