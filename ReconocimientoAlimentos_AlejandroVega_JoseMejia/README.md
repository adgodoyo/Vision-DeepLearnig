# Sistema de Visión Computacional para Control de Inventario Doméstico

### Alejandro Vega - Jose Mejia

---

## 1. Resumen del problema y su impacto social

Cada año se desperdician toneladas de alimentos en hogares por falta de control sobre lo que hay en la nevera o la despensa. Muchos productos expiran sin ser consumidos, lo que genera un impacto económico y ambiental negativo.

Este proyecto busca mitigar ese problema mediante un sistema de visión por computador que:

- Detecta automáticamente productos en la nevera;
- Indica qué tan llena está la nevera;
- Permite llevar un control visual y actualizado del inventario doméstico, recomendando compras;
- Permite detectar productos en el supermercado en tiempo real (cámara) o sobre un video, para marcar como adquiridos los productos faltantes.

**Enfoque social:**  
Este sistema busca reducir el desperdicio de alimentos, ahorrar dinero en compras innecesarias y fomentar el consumo responsable en el hogar. Puede aplicarse en casas, residencias estudiantiles, comedores comunitarios o incluso en tiendas.

---

## 2. Descripción de la arquitectura y desarrollo por fases

El sistema está compuesto por un pipeline que integra dos grandes módulos:

### Fase 1: Inspección de la nevera (imagen estática)

- **Procesamiento de imagen:** Se toma una foto de la nevera y se ecualiza el histograma para compensar diversas condiciones de luz.
- **Extracción de características:** Un modelo ResNet50 procesa la imagen y, mediante comparación de embeddings, estima el porcentaje de ocupación (llenado) de la nevera respecto a referencias de vacío y lleno.
- **Detección de productos:** Un modelo YOLO identifica los productos presentes dentro de una lista predefinida.
- **Resultado:** El sistema presenta los productos encontrados en la nevera, indica aquellos por comprar (de la lista objetivo) y muestra un estimado del porcentaje de llenado.

### Fase 2: Detección en supermercado (cámara o video)

- **Funcionalidad:** El sistema abre la cámara web para detectar productos faltantes en la nevera mientras estás en el supermercado. Si la cámara no está disponible o así se define, utiliza un video de ejemplo para la detección.
- **Tecnología:** Se emplea OpenCV para la captura en tiempo real/video y el mismo modelo YOLO para identificar productos.
- **Proceso:** Por cada producto detectado que faltaba en la nevera, lo retira de la lista de compras en tiempo real.
- **Visualización:** Se presentan en pantalla los productos reconocidos en video/cámara, junto con mensajes indicando cuáles aún falta comprar.

---

## 3. Detalle de los datasets

| Dataset                   | Tipo   | Fuente                   | Descripción                                              |
|---------------------------|--------|--------------------------|----------------------------------------------------------|
| Productos de supermercado | Propio | Fotografías del grupo    | Imágenes de estantes reales en tiendas y supermercados   |
| Productos en la nevera    | Propio | Fotografías del grupo    | Imágenes con fondo doméstico de productos reales         |

El dataset fue recolectado por los autores, etiquetado en formato YOLO con clases como 'leche', 'atún', 'mayonesa', 'queso parmesano', entre otros.

---

## Ejecución del pipeline en local

### Estructura esperada:
```
/ReconocimientoAlimentos_AlejandroVega_JoseMejia/
├── run_pipeline.py
├── requirements.txt
├── data/
│ ├── nevera_prueba.jpg
│ └── video_prueba.mp4
├── src/
│ ├── runs/embedding_lleno.npy
│ ├── runs/embedding_vacio.npy
│ ├── runs/resnet50_feature_extractor.pth
│ └── runs/detect/train/weights/best.pt
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

3. El sistema analizará la imagen de tu nevera, mostrará los productos que hay y su llenado estimado.  
   Luego, intentará abrir la cámara web para detectar productos en el supermercado (o puedes ajustar para usar un video de ejemplo). Pulsa 'q' para salir de la detección en tiempo real.

    - Si la cámara no está disponible o así lo defines (`usar_video=True`), el sistema procesará el archivo de video definido en `video_path`.

---

## 4. Lecciones aprendidas y trabajo futuro

### Lecciones aprendidas:
- La calidad del dataset propio es esencial para la precisión del modelo.
- La integración de diferentes modelos (YOLO + ResNet50) en un solo pipeline permite obtener métricas cuantitativas y listas accionables para el usuario.
- La visualización en tiempo real requiere una correcta gestión de recursos y una interfaz comprensible para el usuario.

### Trabajo futuro:
- Incluir lectura de fechas de vencimiento por OCR.
- Implementar un sistema de notificaciones para productos próximos a vencer o faltantes.
- Mantener historial de entradas/salidas con tracking visual o matching.
- Desarrollar una interfaz de usuario más amigable y multiplataforma.

---
