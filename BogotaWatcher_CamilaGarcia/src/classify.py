import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
import os

def entrenar_clasificador(
    data_dir="data/clasificador",
    modelo_salida="modelo_clasificador.h5",
    epochs=10
):
    """
    Entrena un clasificador de imágenes usando MobileNetV2 como base.

    Parámetros:
    - data_dir (str): Directorio con subcarpetas por clase que contienen imágenes.
    - modelo_salida (str): Ruta donde se guardará el modelo entrenado (.h5).
    - epochs (int): Número de épocas de entrenamiento.

    Flujo interno:
    1. Define tamaño de imagen y tamaño de batch.
    2. Crea un ImageDataGenerator para normalizar y partir en entrenamiento/validación.
    3. Construye generadores (train_gen y val_gen) desde el directorio.
    4. Carga MobileNetV2 preentrenado y congela sus capas.
    5. Añade pooling global y capa densa final acorde al número de clases.
    6. Compila el modelo con Adam y pérdida de entropía cruzada categórica.
    7. Ajusta el modelo con los generadores y guarda el resultado.
    """

    # 1. Parámetros de imagen y batch
    img_size = 224
    batch_size = 16

    # 2. Generador de datos: reescala píxeles a [0,1] y reserva 20% para validación
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    # 3a. Generador de entrenamiento
    train_gen = datagen.flow_from_directory(
        data_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'   # usa el 80% de las imágenes
    )

    # 3b. Generador de validación
    val_gen = datagen.flow_from_directory(
        data_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation' # usa el 20% restante
    )

    # 4. Carga MobileNetV2 sin la parte superior (include_top=False)
    base_model = MobileNetV2(
        weights='imagenet', 
        include_top=False, 
        input_shape=(img_size, img_size, 3)
    )
    base_model.trainable = False  # 4b. Congelar pesos preentrenados

    # 5. Construcción de la cabeza de clasificación
    x = base_model.output
    x = GlobalAveragePooling2D()(x)  # pooling global reduce cada mapa de características a un valor
    predictions = Dense(
        len(train_gen.class_indices),  # tantas neuronas como clases
        activation='softmax'           # salida de probabilidades
    )(x)

    # Ensamblar modelo final
    model = Model(inputs=base_model.input, outputs=predictions)

    # 6. Compilar con optimizador Adam y pérdida adecuada para clasificación múltiple
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # 7a. Inicio del entrenamiento
    print("🚀 Iniciando entrenamiento...")
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs
    )

    # 7b. Guardar modelo entrenado en disco
    print(f"💾 Guardando modelo en: {modelo_salida}")
    model.save(modelo_salida)

    return model


if __name__ == "__main__":
    entrenar_clasificador(
        data_dir="data/clasificador",
        modelo_salida="modelo_clasificador.h5",
        epochs=10
    )
