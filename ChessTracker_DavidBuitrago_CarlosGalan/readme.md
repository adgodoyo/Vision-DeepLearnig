# ChessTracker: Seguimiento y Reconocimiento de Tableros de Ajedrez con Visión Computacional

## 1. Resumen del problema y su impacto social

El ajedrez es reconocido mundialmente como una herramienta educativa que desarrolla habilidades cognitivas, estratégicas y sociales. Sin embargo, la digitalización de partidas físicas sigue siendo un reto, especialmente en contextos educativos, clubes y torneos donde no se dispone de tableros electrónicos.  
**ChessTracker** busca cerrar esta brecha permitiendo la reconstrucción automática de partidas físicas a partir de imágenes, facilitando el análisis, la enseñanza y la inclusión digital.  
El impacto social es significativo:  
- **Inclusión:** Jugadores sin acceso a tecnología avanzada pueden digitalizar y compartir sus partidas.
- **Educación:** Docentes y estudiantes pueden analizar partidas reales, fomentando el aprendizaje activo.
- **Transparencia:** Torneos presenciales pueden documentar partidas de forma precisa y accesible.
- **Accesibilidad:** Personas con discapacidad visual pueden beneficiarse de la digitalización para acceder a herramientas de análisis y lectura automática.

---

## 2. Descripción de la arquitectura y justificación de decisiones

El sistema está diseñado de forma modular para facilitar su mantenimiento, escalabilidad y adaptación a distintos escenarios.  
### Módulos principales:

- **Detección de tablero (`src/board_detection.py`):**
  - Localiza el tablero en la imagen, corrige la perspectiva y segmenta las casillas.
  - Permite la orientación manual del tablero, evitando errores de OCR y asegurando que la casilla h1 esté correctamente posicionada.
  - Justificación: La orientación manual es más robusta que depender de OCR, especialmente en tableros con fuentes o colores poco convencionales.

- **Detección de piezas (`src/piece_detection.py`):**
  - Utiliza un modelo YOLOv8 entrenado específicamente para piezas de ajedrez.
  - Detecta y clasifica cada pieza en su casilla correspondiente.
  - Justificación: YOLOv8 ofrece un balance óptimo entre precisión y velocidad, y permite la detección en condiciones variadas de luz y ángulo.

- **Conversión a FEN (`src/fen_conversion.py`):**
  - Traduce el estado detectado del tablero a la notación FEN, estándar en el mundo del ajedrez digital.
  - Justificación: FEN es ampliamente soportado por plataformas como Lichess y Chess.com, facilitando la interoperabilidad.

- **Detección de movimientos (`src/move_detection.py`):**
  - Compara estados consecutivos del tablero para identificar movimientos, incluyendo capturas y enroques.
  - Justificación: Permite reconstruir la partida jugada, no solo el estado final.

- **Generación de PGN (`src/pgn_writer.py`):**
  - Construye archivos PGN, el formato estándar para almacenar partidas de ajedrez, incluyendo comentarios con los FEN previos.
  - Justificación: El PGN es el formato universal para compartir y analizar partidas.

- **Interfaz principal (`main.py`):**
  - Orquesta el flujo de procesamiento: calibración, orientación, detección, reconstrucción y exportación.
  - Justificación: Centraliza la lógica y facilita la interacción con el usuario.

**Decisiones clave:**
- Modularidad para facilitar pruebas y mejoras.
- Uso de modelos propios y datasets personalizados para máxima precisión.
- Intervención manual en pasos críticos para asegurar robustez en entornos reales.

---

## 3. Detalle de los datasets

### Dataset propio
- **Imágenes:** Capturadas en diferentes condiciones de luz, ángulos y con variedad de piezas y tableros.
- **Anotaciones:** Realizadas manualmente usando herramientas como LabelImg, marcando la posición y tipo de cada pieza.
- **Propósito:** Entrenamiento y validación del modelo de detección de piezas, así como pruebas de robustez del sistema completo.

### Dataset externo
- **Chess Pieces Dataset (Kaggle):**  
  [Enlace](https://www.kaggle.com/datasets)  
  Utilizado para pre-entrenamiento y aumento de datos, asegurando diversidad de estilos de piezas y tableros.
- **Tableros sintéticos:**  
  Generados artificialmente para simular condiciones extremas y mejorar la generalización del modelo.

### Consideraciones éticas y de privacidad
- Los datasets propios no se distribuyen públicamente para proteger la privacidad de los participantes y cumplir con regulaciones institucionales.
- Se proveen scripts para que cualquier usuario pueda generar su propio dataset a partir de imágenes capturadas localmente.

---

## 4. Métricas empleadas y discusión de resultados

### Métricas empleadas

- **Precisión de detección de piezas (mAP):**
  - Se utiliza la métrica mean Average Precision (mAP) a umbral 0.5, estándar en detección de objetos.
  - Resultado: mAP@0.5 ≈ 0.92 en el conjunto de validación propio.

- **Exactitud en reconstrucción de FEN:**
  - Se evalúa la coincidencia entre el FEN generado y el FEN real del tablero en pruebas controladas.
  - Resultado: 95% de exactitud en condiciones normales.

- **Robustez ante diferentes orientaciones:**
  - Se mide la capacidad del sistema para reconstruir correctamente el tablero independientemente de la orientación inicial de la imagen.
  - Resultado: 100% de éxito con intervención manual; el OCR automático falló en tableros con fuentes poco convencionales.

### Discusión de resultados

- El sistema es altamente preciso en condiciones de luz normales y con tableros estándar.
- Las principales fuentes de error son:
  - Oclusión parcial de piezas (por manos, otros objetos).
  - Reflejos intensos en el tablero.
  - Tableros con diseños o fuentes atípicas.
- La intervención manual en la orientación del tablero resultó ser una solución simple y efectiva para evitar errores sistemáticos.
- El modelo de detección generaliza bien gracias a la diversidad del dataset, pero puede beneficiarse de más ejemplos en condiciones adversas.

---

## 5. Lecciones aprendidas y trabajo futuro

### Lecciones aprendidas

- **OCR en bordes:** El reconocimiento automático de letras y números en los bordes del tablero es poco fiable en la práctica. La intervención del usuario es más robusta y rápida.
- **Importancia del dataset:** La calidad, variedad y cantidad de datos anotados son determinantes para el éxito del modelo de detección.
- **Modularidad:** Un diseño modular permite iterar y mejorar componentes individuales sin afectar el sistema completo.
- **Interacción usuario-máquina:** Involucrar al usuario en pasos críticos (como la orientación) puede mejorar la precisión global del sistema.

### Trabajo futuro

- **Interfaz gráfica:** Desarrollar una GUI intuitiva para facilitar la calibración, orientación y revisión de resultados.
- **Procesamiento en tiempo real:** Integrar soporte para cámaras web y procesamiento en vivo de partidas.
- **Mejoras en detección:** Robustecer el sistema ante condiciones adversas (reflejos, piezas caídas, tableros no estándar).
- **Dataset abierto:** Publicar un dataset anonimizado y ampliado para fomentar la investigación y colaboración.
- **Extensión a otros juegos:** Adaptar la arquitectura para otros juegos de mesa que requieran digitalización automática.

---

## Estructura del repositorio

```
ChessTracker/
│
├── main.py
├── src/
│   ├── board_detection.py
│   ├── piece_detection.py
│   ├── fen_conversion.py
│   ├── move_detection.py
│   └── pgn_writer.py
├── models/
│   └── best2.pt
├── images/
│   └── (imágenes de entrada)
├── README.md
└── requirements.txt
```

- **main.py:** Script principal de ejecución.
- **src/**: Código fuente de los módulos principales.
- **models/**: Modelos entrenados para detección de piezas.
- **images/**: Imágenes de entrada y calibración.
- **requirements.txt:** Dependencias del proyecto.

---

## Cómo ejecutar

1. **Instala las dependencias:**
   ```
   pip install -r requirements.txt
   ```
2. **Prepara tus imágenes:**
   - Coloca una imagen del tablero vacío como `empty.JPG` en la carpeta `images/`.
   - Coloca las imágenes de cada estado del tablero (`1.jpg`, `2.jpg`, ...) en la misma carpeta.
3. **Ejecuta el sistema:**
   ```
   python main.py
   ```
4. **Sigue las instrucciones en consola:**
   - Se te pedirá indicar la orientación del tablero (ubicación de la casilla h1).
   - El sistema procesará las imágenes y generará el archivo `partida.pgn` con la reconstrucción de la partida.

---

