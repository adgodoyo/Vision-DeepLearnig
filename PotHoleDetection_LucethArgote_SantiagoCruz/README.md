# 🚧 Detección de Huecos en Vía Pública con YOLOv8 🚦

## 1. Resumen del problema y su impacto social

### 1.1. ¿Cuál es el problema? 🤔

¡Los huecos y baches en las calles son un dolor de cabeza! No solo dañan los carros 🚗, sino que pueden causar accidentes y hacer que el tráfico sea un caos. El problema es que muchas veces solo se arreglan cuando alguien se queja, y eso puede tomar mucho tiempo.

Por eso, este proyecto usa visión por computadora para detectar huecos automáticamente usando imágenes de calles (de Google Maps, videos y un dataset externo). Así, las ciudades pueden saber dónde están los problemas sin esperar a que alguien los reporte.

### 1.2. ¿Por qué importa? 🌎

Tener una herramienta automática para detectar huecos ayuda a que las ciudades sean más seguras y eficientes. Las autoridades pueden arreglar los problemas más rápido, hay menos accidentes y los recursos se usan mejor. Además, hasta las ciudades pequeñas pueden monitorear sus calles sin tener que salir a buscar huecos uno por uno. ¡Ideal para ciudades inteligentes! 🏙️

---

## 2. ¿Cómo lo hicimos? Arquitectura y decisiones técnicas 🛠️

### 2.1. ¿Qué modelo usamos? 🤖

Usamos **YOLOv8s-Segmentation** (YOLOv8-seg), que no solo detecta objetos, ¡también los dibuja pixel a pixel! Esto es clave para saber exactamente dónde empieza y termina un hueco.

El proceso fue así:
1. Juntamos y etiquetamos imágenes 📸
2. Entrenamos el modelo con esas imágenes
3. Evaluamos el modelo con métricas estándar 📊
4. Probamos el modelo en imágenes nuevas para ver qué tal funcionaba

Durante el entrenamiento, "congelamos" el backbone del modelo (las capas que ven cosas generales como bordes y texturas), así el modelo aprende más rápido y con menos datos.

### 2.2. ¿Por qué YOLOv8-seg? 💡

- **Fácil de usar**: Tiene una interfaz amigable y modular.
- **Rápido**: Perfecto para aplicaciones en tiempo real.
- **Ya viene preentrenado**: Así que no necesitamos tantos datos ni tanto tiempo para entrenar.

---

## 3. Sobre los datasets 📂

### 3.1. Dataset propio

Agregamos **4 imágenes de Google Maps** y algunos fotogramas de un **video de YouTube**. Todas muestran calles en el día y fueron etiquetadas a mano.

### 3.2. Dataset externo

El dataset principal viene del paper de Chu et al. (2023): **51 imágenes de huecos** en calles, desde varios ángulos. Las etiquetamos manualmente para que funcionaran con YOLOv8-seg.

### 3.3. ¿Cómo etiquetamos y preprocesamos? 🖍️

Usamos **Roboflow** para segmentar solo los huecos (pensamos en agregar grietas, pero era mucho trabajo). También hicimos aumentos de datos (rotaciones, volteos, etc.)

---

## 4. Métricas y resultados 📈

### 4.1. ¿Cómo medimos el desempeño?

- **mAP@0.5**: Qué tan bien detecta huecos con al menos 50% de superposición.
- **mAP@0.5:0.95**: Igual, pero más exigente (varios umbrales).
- **Precisión**: Cuántos de los huecos detectados realmente son huecos.

### 4.2. ¿Qué tal salió? 🚀

¡Bastante bien! El modelo detecta la mayoría de los huecos con buena precisión y superposición.

### 4.3. ¿Qué podría mejorar?

A veces se confunde con sombras o manchas oscuras (falsos positivos). Así que todavía hay espacio para mejorar, sobre todo en condiciones de luz difíciles.

---

## 5. Lecciones y futuro 🌱

### 5.1. ¿Qué aprendimos?

Entrenar un modelo de segmentación hoy en día es mucho más fácil de lo que parece. Con pocos datos y las herramientas correctas, se pueden lograr cosas útiles. Además, ¡la IA puede ayudar a resolver problemas reales de la ciudad!

### 5.2. ¿Qué sigue?

- Agregar más imágenes (de noche, con lluvia, etc.)
- Incluir otras clases (grietas, obstáculos, etc.)
- Probar el modelo en video para monitoreo en tiempo real 🎥
- Agregar geolocalización para mapear los huecos en un mapa 🗺️

---

## 6. Referencias 📚

Chu, H., Saeed, M.R., Rashid, J., Mehmood, M.T., Ahmad, I. et al. (2023).  
*Deep Learning Method to Detect the Road Cracks and Potholes for Smart Cities.*  
Computers, Materials & Continua, 75(1), 1863–1881.  
https://doi.org/10.32604/cmc.2023.035287