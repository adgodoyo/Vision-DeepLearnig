
# 🏒 Proyecto Final: Sistema de Visión Computacional para Análisis de Seguridad y Jugadas en Hockey

### Isabela Ruiz - Natalia Cabrera

---

## 1. 🧩 Resumen del problema y su impacto social

El hockey en linea  es un deporte de alto impacto físico. Los jugadores se enfrentan a velocidades altas, superficies duras, y contacto frecuente con otros jugadores o el puck. A pesar de la exigencia reglamentaria del uso de casco y guantes, en contextos escolares o ligas menores muchas veces esta protección no se usa correctamente, o no se verifica.

Este proyecto busca **mejorar la seguridad y el monitoreo automatizado** de los partidos, a través de una herramienta que:

- Detecta y sigue jugadores en video automáticamente.
- Verifica si cada jugador usa casco y guantes.
- Clasifica jugadas clave como goles, tiros o pases.
- Genera un video final anotado con IDs, equipo de protección y alertas visuales.

### 💡 Enfoque social:
Este sistema tiene como objetivo contribuir a la prevención de lesiones y a la formación de una cultura de seguridad en el deporte. Es aplicable a ligas juveniles, escuelas, torneos aficionados o prácticas profesionales, promoviendo buenas prácticas y reduciendo el trabajo manual de análisis de grabaciones.

---

## 2. 🧠 Descripción de la arquitectura y desarrollo por fases

El sistema fue desarrollado en **cuatro fases principales**, todas con modelos entrenados y evaluados de forma independiente, y luego integradas en un pipeline robusto:

### 🔹 Fase 1: Detección de jugadores y puck

- **Modelo:** YOLOv8 (versión pequeña) entrenado desde cero con imágenes etiquetadas por el grupo.
- **Desarrollo:** Se construyó un dataset balanceado con anotaciones YOLO, se entrenó por 30 épocas y se evaluó con mAP@0.5.
- **Resultado:** El modelo detecta correctamente jugadores en la mayoría de los frames, incluso con movimiento o cambios de cámara.
- **Aplicación:** Bounding boxes sobre cada jugador y puck en los videos.

### 🔹 Fase 2: Tracking de jugadores

- **Técnica:** DeepSORT (algoritmo de tracking multiobjeto basado en detección + movimiento).
- **Desarrollo:** Se adaptó el código de SORT para integrarse con las predicciones de YOLOv8 en cada frame.
- **Resultado:** Se asignan IDs únicos a cada jugador, visibles durante todo el video.
- **Visualización:** Mapa de calor de trayectorias, conteo por ID, continuidad temporal.

### 🔹 Fase 3: Clasificación de jugadas

- **Modelo:** Red neuronal simple entrenada sobre clips de video (2–3 segundos) clasificados como `gol`, `tiro`, `pase`.
- **Desarrollo:** Se dividieron los videos, se etiquetaron manualmente, se entrenó la red y se integró un clasificador por clip.
- **Resultado:** Jugadas clave son clasificadas con buena precisión (>85%) y anotadas en el video.

### 🔹 Fase 4: Verificación del equipo de protección

- **Modelos:** Dos clasificadores ResNet18 reentrenados para:
  - Casco vs. sin casco.
  - Guantes vs. sin guantes.
- **Desarrollo:** Se creó un dataset con imágenes recortadas de jugadores, se entrenaron los clasificadores en PyTorch.
- **Resultado:** El sistema detecta visualmente si el jugador cumple con la seguridad.
- **Visualización:** Etiquetas ✔ o 🚫 sobre cada jugador detectado.

---

## 3. 🗂️ Detalle de los datasets

| Dataset                      | Tipo        | Fuente                | Descripción                                                |
|-----------------------------|-------------|------------------------|------------------------------------------------------------|
| Detección jugadores/puck    | Propio      | Etiquetado manual      | Imágenes de partidos reales etiquetadas con formato YOLO   |
| Seguimiento (tracking)      | Derivado    | A partir de detección  | Videos procesados con YOLO + DeepSORT                      |
| Clasificación de jugadas    | Propio      | Videos divididos       | Clips de 2–3 segundos anotados manualmente                 |
| Casco y guantes             | Propio      | Dataset recolectado    | Imágenes clasificadas manualmente por tipo de protección   |

- Archivos omitidos por limitaciones de GitHub

Este proyecto utiliza más de 300 imágenes clasificadas y más de 70 clips de video para entrenamiento, validación y pruebas en las distintas fases (detección, clasificación de jugadas, y verificación de equipo de protección). Sin embargo, por restricciones de GitHub:

No se incluyeron todos los archivos multimedia en este repositorio.

En particular, no se subieron los siguientes:

- Videos completos del dataset (clips de jugadas).

- Imágenes clasificadas para casco y guantes (más de 300 en total).

- Imágenes de validación 

---

## 4. 📈 Métricas y resultados

| Tarea                        | Modelo         | Métrica clave                | Resultado aproximado |
|-----------------------------|----------------|------------------------------|-----------------------|
| Detección de jugadores      | YOLOv8         | mAP@0.5                      | ~91%                  |
| Seguimiento (tracking)      | DeepSORT       | IDF1                         | Alta consistencia     |
| Clasificación de jugadas    | Red personalizada | Accuracy en validación    | ~85%                  |
| Casco / Guantes             | ResNet18       | Accuracy por clase           | >90%                  |

---

## 🚀 ¿Cómo ejecutar el pipeline en Google Colab?

### Estructura esperada:

```
/content/
├── run_pipeline.py
├── requirements.txt
├── data/test.mp4
├── models/best.pt
├── models/modelo_casco.pth
├── models/modelo_guantes.pth
├── src/sort.py
```

### Pasos:

1. Instala dependencias:

```bash
!pip install -r requirements.txt
```

2. Ejecuta:

```bash
!python run_pipeline.py
```

3. Resultado por frame:

```bash
🟦 Frame 12:
🎯 Detecciones: 2
   🧍 ID 0 → Casco: ✔ | Guantes: ✔
   🧍 ID 1 → Casco: ❌ | Guantes: ✔
```

### ✅ Ejemplo real de salida final:

```
======================================
📊 RESUMEN GENERAL DEL VIDEO
======================================
🔢 Jugadores detectados (IDs): [0, 1, 2, ..., 502]
🚨 Jugadores sin casco: [0, 1, 2, 6, 8, ..., 491]
🚨 Jugadores sin guantes: [7, 9, 47, ..., 491]

📥 Descargando video procesado...
```

---

## 5. 🧪 Lecciones aprendidas y trabajo futuro

### Lecciones aprendidas:

- Entrenar con tus propios datos requiere tiempo pero mejora la personalización.
- La modularidad (fases separadas) permite depurar fácilmente errores.
- La integración de detección, seguimiento y clasificación produce una solución completa y visualmente clara.

### Trabajo futuro:

- Agregar segmentación de cancha para analizar zonas activas.
- Detectar eventos anómalos como caídas, peleas o lesiones.
- Mejorar clasificación de jugadas con modelos temporales (CNN+LSTM).
- Desplegar en tiempo real con Jetson Nano o Raspberry Pi.
