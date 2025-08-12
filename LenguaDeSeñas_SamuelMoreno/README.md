# Clasificación de Lengua de Señas Colombiana

## _Resumen del problema_

Actualmente existe una gran falta de comunicación efectiva entre personas oyentes y no oyentes que utilizan la Lengua de Señas Colombiana (LSC). Esta situación genera barreras significativas que limitan la interacción social, el acceso a servicios, educación y oportunidades laborales para las personas no oyentes. La dependencia de intérpretes, que no siempre están disponibles, agrava esta problemática, aumentando la frustración y el aislamiento social. En este contexto, surge la necesidad de una herramienta tecnológica accesible que facilite la comunicación en tiempo real, promoviendo así la inclusión y mejorando la calidad de vida de la comunidad no oyente en Colombia.

Es por eso que en este proyecto se busca desarrollar una arquitectura basada en *deep learning* que sea capaz de reconocer señas específicas de la Lengua de Señas Colombiana a partir de imágenes. Esta herramienta podría marcar el inicio de una serie de proyectos que faciliten la comunicación entre personas sordas y oyentes en entornos donde no hay intérpretes, mejorando el acceso a la información en tiempo real y fomentando la autonomía de la comunidad sorda.

## _Descripción de la arquitectura_

- **Extracción de características:** La base de la arquitectura es la red *MobileNetV2*, la cual ya está entrenada para clasificación de imágenes con muchas categorías. Sin embargo, en este proyecto solo se usaron las capas que extraen las características para poderlas conectar con redes posteriores.
  
- **Localización:** Se agregaron dos capas densas y una de Dropout para reducir el sobreajuste. Luego de estas, la última capa contiene las cuatro neuronas que corresponden a la salida de la regresión de los valores del cuadrado que encierra la mano.

- **Segmentación:** En esta parte se volvió a usar la *MobileNet* base, pero con salidas diferentes. En este caso, lo que se usó fueron las salidas de algunas capas luego de las activaciones ReLU como *encoder*. Para el *decoder*, se usaron los bloques de *umsample* que ya están implementados en los ejemplos que se encuentran en el repositorio de *TensorFlow*. La salida de este modelo es de tamaño $128\times128\times2$, ya que se tienen dos posibles categorías para cada pixel en la imagen.

- **Clasificación:** De nuevo, se usó la red *MobileNet* para extraer las características de la mano dentro de la imagen ya recortada (localización) y sin el fondo (segmentación), las cuales pasaron a través de unas capas densas y una Dropout. La diferencia con la primera sección es la salida, que en este caso contiene cinco neuronas y una función de activación *softmax* para retornar un vector de probabilidaddes sobre las clases de señas.

## _Datos utilizados_

Inicialmente, se recopilaron $30$ fotos de cada seña por medio de *OpenCV*, para un total de 145 fotos (solo se guardaron 25 fotos de la seña "Adiós"). Para la tarea de localización, se utilizó [Label Studio](https://labelstud.io/) para dibujar los cuadros y obtener las [coordenadas](https://github.com/adgodoyo/Vision-DeepLearnig/tree/grupo14_SamuelMoreno/LenguaDeSe%C3%B1as_SamuelMoreno/data/localization) requeridas para el entrenamiento en formato CSV.

Luego, para el modelo de segmentación, se seleccionaron $16$ imágenes de cada categoría y se recortaron dejando solamente los cuadros de las manos. Estas imágenes recortadas se [colorearon](https://github.com/adgodoyo/Vision-DeepLearnig/tree/grupo14_SamuelMoreno/LenguaDeSe%C3%B1as_SamuelMoreno/data/segmentation) para que la mano quedara totalmente blanca y el fondo quedara negro (lo que sería la máscara objetivo durante el entrenamiento). Estas máscaras ideales también se usaron para crear el conjunto de datos que se pasaría al modelo de clasificación final, el cual consistió en las mismas imágenes recortadas pero sin el fondo.

## _Discusión de resultados_

La conjunción de los tres modelos obtuvo un rendimiento relativamente satisfactorio al clasificar las señas seleccionadas. Al final se evidenció que el modelo aún comete errores con algunas señas específicas, pero puede deberse a la primera parte de la arquitectura, correspondiente a la localización de la mano.

Otro aspecto importante es el tiempo de inferencia durante el *pipeline*, ya que cada imagen se demora 9 segundos en procesarse. Aun así, en el [notebook](https://github.com/adgodoyo/Vision-DeepLearnig/blob/grupo14_SamuelMoreno/LenguaDeSe%C3%B1as_SamuelMoreno/src/Taller_3.ipynb) se evidencia que el modelo clasifica exitosamente varias imágenes.

## _Lecciones aprendidas y trabajo futuro_

El tiempo de ejecución podría reducirse un poco limitando la arquitectura a la localización y clasificación sin pasar por un proceso de segmentación de las manos. Esto sería útil para permitir una detección en tiempo real y/o procesamiento de video, ya que la gran mayoría de las señas en este lenguaje se forman con movimientos y gestos de la cara.

## _Referencias_

- [Diccionario Básico de la Lengua de Señas Colombiana](https://www.colombiaaprende.edu.co/sites/default/files/files_public/2022-04/Diccionario-lengua-de-senas.pdf)
- [Object Localization with Tensorflow](https://github.com/aliosmankaya/ObjectLocalizationWithTensorflow)
- [TensorFlow Examples](https://github.com/tensorflow/examples)
- [TensorFlow - Image segmentation](https://www.tensorflow.org/tutorials/images/segmentation)
- [tf.keras.applications.MobileNetV2](https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV2)
