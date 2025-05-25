import pickle, numpy as np, tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.utils import to_categorical

from model import get_model
from constants import *

def training_model(model_path: str, epochs: int = 200):
    # ── 1. Carga conjuntos ya preparados ────────────────────────────
    with open(os.path.join(DATA_PATH, "train_val_test.pkl"), "rb") as f:
        data = pickle.load(f)

    X_train, y_train = data["X_train"], to_categorical(data["y_train"])
    X_val,   y_val   = data["X_val"],   to_categorical(data["y_val"])
    word_ids         = data["word_ids"]            # lista de etiquetas

    # ── 2. Pesos de clase (por si hay leve desequilibrio) ───────────
    class_weight = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(data["y_train"]),
        y=data["y_train"]
    )
    class_weight = {i: w for i, w in enumerate(class_weight)}

    # ── 3. Modelo ───────────────────────────────────────────────────
    model = get_model(MODEL_FRAMES, len(word_ids))
    model.summary()

    callbacks = [
        ReduceLROnPlateau(factor=0.5, patience=5),
        EarlyStopping(patience=15, restore_best_weights=True),
        ModelCheckpoint(model_path, save_best_only=True)
    ]

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=32,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1
    )

    # Guarda también el modelo final (opcional)
    model.save(model_path)

if __name__ == "__main__":
    training_model(MODEL_PATH)
