# 🎙️🤟 Traductor LSC → Voz en Tiempo Real

Sistema completo que **detecta, reconoce y vocaliza**  
diez gestos básicos de la **Lengua de Señas Colombiana (LSC)**  
empleando únicamente una cámara web y la **CPU**.

<p align="center">
  <img src="Figures/demo_gui.gif" width="600">
</p>

---

## 📊 Resultados

<table>
  <thead>
    <tr>
      <th style="text-align:left;">Conjunto</th>
      <th style="text-align:center;">Accuracy</th>
      <th style="text-align:center;">Macro-F<sub>1</sub></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Test</strong></td>
      <td style="text-align:center;">0.981</td>
      <td style="text-align:center;">0.981</td>
    </tr>
  </tbody>
</table>

---

## 🗂️ Estructura del repositorio

```plaintext
.
├─ capture_samples.py      # Paso 1 – Grabación automática
├─ normalize_samples.py    # Paso 2 – Interpolación a 15 frames
├─ create_keypoints.py     # Paso 3 – Extracción de 1 662 key-points
├─ prepare_dataset.py      # Paso 4 – Split estratificado 70/15/15
├─ model.py                # Paso 5 – Red TCN + Attention
├─ training_model.py       # Paso 6 – Entrenamiento
├─ confusion_matrix.py     # Paso 7 – Métricas y gráficas
├─ plot_pr_curves.py       # Extensión: curvas PR
├─ latent_tsne_umap.py     # Extensión: t-SNE / UMAP
├─ main.py                 # Paso 8 – GUI PyQt5 en tiempo real
├─ text_to_speech.py       # Paso 9 – Síntesis de voz
├─ data/                   # Key-points y splits serializados
├─ frame_actions/          # Frames JPG por gesto
└─ models/                 # Modelo *.keras* y words.json


---

## 🗂️ Estructura del repositorio

```

.
├─ capture\_samples.py      # Paso 1 – Grabación automática

├─ normalize\_samples.py    # Paso 2 – Interpolación a 15 frames

├─ create\_keypoints.py     # Paso 3 – Extracción de 1 662 key-points

├─ prepare\_dataset.py      # Paso 4 – Split estratificado 70/15/15

├─ model.py                # Paso 5 – Red TCN + Attention

├─ training\_model.py       # Paso 6 – Entrenamiento

├─ confusion\_matrix.py     # Paso 7 – Métricas y gráficas

├─ plot\_pr\_curves.py       # Extensión: curvas PR

├─ latent\_tsne\_umap.py     # Extensión: t-SNE / UMAP

├─ main.py                 # Paso 8 – GUI PyQt5 en tiempo real

├─ text\_to\_speech.py       # Paso 9 – Síntesis de voz

├─ data/                   # Key-points y splits serializados

├─ frame\_actions/          # Frames JPG por gesto

└─ models/                 # Modelo *.keras* y words.json

````

---

## ⚡ Instalación rápida

```bash
# 1 Crear y activar entorno virtual
python -m venv lsc_env
# Windows ➜ lsc_env\Scripts\activate
source lsc_env/bin/activate

# 2 Instalar dependencias (CPU-only)
pip install -r requirements.txt
#   – TensorFlow 2.16.1
#   – mediapipe
#   – opencv-python
#   – PyQt5
#   – gTTS, pygame
#   – seaborn, umap-learn …

# 3 Descargar modelo pre-entrenado
curl -L -o models/actions_15.keras https://…/actions_15.keras
````

> El sistema corre a \~25 fps en un portátil i5; **no se requiere GPU**.

---

## ▶️ Demo en vivo

```bash
python main.py
```

1. Se abre la webcam.
2. Realice un gesto completo; la aplicación lo detecta automáticamente.
3. Al acabar, se muestra el texto y se reproduce la voz en castellano.

---

## 🔬 Entrenamiento desde cero

```bash
# 1 Capturar datos (≈200 muestras por gesto)
python capture_samples.py --word hola

# 2 Normalizar e indexar key-points
python normalize_samples.py
python create_keypoints.py

# 3 Generar splits y entrenar
python prepare_dataset.py
python training_model.py            # produce models/actions_15.keras
```

---

## 📈 Evaluación y gráficos adicionales

```bash
python confusion_matrix.py          # matriz + reporte
python plot_pr_curves.py            # curvas Precision–Recall
python latent_tsne_umap.py          # mapa t-SNE / UMAP
```

---

<h2>✨ Componentes clave del sistema</h2>

<table>
  <thead>
    <tr>
      <th>Módulo</th>
      <th>Tecnología</th>
      <th>Rol principal</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>🧍‍♂️ Detección / Tracking</td>
      <td>MediaPipe</td>
      <td>1 662 landmarks (pose + face + hands)</td>
    </tr>
    <tr>
      <td>⏱️ Normalización temporal</td>
      <td>OpenCV</td>
      <td>Interpolación / muestreo a 15 frames</td>
    </tr>
    <tr>
      <td>🧠 Clasificación secuencial</td>
      <td><strong>TCN + Attention</strong></td>
      <td>RF 31 frames, 3.5 M parámetros, 98% accuracy</td>
    </tr>
    <tr>
      <td>🖥️ Interfaz gráfica</td>
      <td>PyQt5</td>
      <td>Webcam, overlay de keypoints, texto dinámico</td>
    </tr>
    <tr>
      <td>🔊 Síntesis de voz</td>
      <td>gTTS + pygame</td>
      <td>Locución en español con baja latencia</td>
    </tr>
  </tbody>
</table>


---

## 📝 Descripción breve

El proyecto integra **detección corporal, seguimiento temporal,
normalización, red convolucional dilatada con atención escalar y
síntesis de voz** para ofrecer un traductor LSC-a-Audio portátil,
de código abierto y totalmente funcional en CPU.

---

<h2>🧾 Palabras entrenadas en LSC</h2>

<p>A continuación se muestran las <strong>10 palabras</strong> reconocidas por el sistema, junto a su representación en la <strong>Lengua de Señas Colombiana (LSC)</strong>.</p>

<table>
  <thead>
    <tr>
      <th>Palabra</th>
      <th>Imagen en LSC</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>hola</strong></td>
      <td><img src="LenguajeSenas-to-Voz/Data/Figures/Hola.png" width="180"/></td>
    </tr>
    <tr>
      <td><strong>adios</strong></td>
      <td><img src="LenguajeSenas-to-Voz/Data/Figures/Adios.png" width="180"/></td>
    </tr>
    <tr>
      <td><strong>bien</strong></td>
      <td><img src="LenguajeSenas-to-Voz/Data/Figures/Bien.png" width="180"/></td>
    </tr>
    <tr>
      <td><strong>mal</strong></td>
      <td><img src="LenguajeSenas-to-Voz/Data/Figures/Mal.png" width="180"/></td>
    </tr>
    <tr>
      <td><strong>mas_o_menos</strong></td>
      <td><img src="LenguajeSenas-to-Voz/Data/Figures/MasOMenos.png" width="180"/></td>
    </tr>
    <tr>
      <td><strong>lo_siento</strong></td>
      <td><img src="LenguajeSenas-to-Voz/Data/Figures/LoSiento.png" width="180"/></td>
    </tr>
    <tr>
      <td><strong>como_estas</strong></td>
      <td><img src="LenguajeSenas-to-Voz/Data/Figures/ComoEstas.png" width="180"/></td>
    </tr>
    <tr>
      <td><strong>feliz_dia</strong></td>
      <td><img src="LenguajeSenas-to-Voz/Data/Figures/FelizDia.png" width="180"/></td>
    </tr>
    <tr>
      <td><strong>familia</strong></td>
      <td><img src="LenguajeSenas-to-Voz/Data/Figures/Familia.png" width="180"/></td>
    </tr>
    <tr>
      <td><strong>Papa</strong></td>
      <td><img src="LenguajeSenas-to-Voz/Data/Figures/Papa.png" width="180"/></td>
    </tr>
  </tbody>
</table>

---

## 📚 Créditos

**Autores:**  Juan Sebastian Rodriguez Salazar y Johan Santiago Caro Valencia / Grupo – Visión Computacional, 2025-I

Agradecimientos al repositorio base [@ronvidev](https://github.com/ronvidev/modelo_lstm_lsp)
por la lógica original de captura.

```
```
