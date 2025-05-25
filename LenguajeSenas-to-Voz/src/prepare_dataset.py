# prepare_dataset.py
import numpy as np, os, pickle
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from helpers import get_word_ids, get_sequences_and_labels
from constants import *

def main(test_size=0.15, val_size=0.15):
    word_ids = get_word_ids(WORDS_JSON_PATH)
    seqs, labels = get_sequences_and_labels(word_ids)

    # Padding antes del split  ✅
    seqs = pad_sequences(
        seqs, maxlen=MODEL_FRAMES,
        padding='pre', truncating='post', dtype='float32'
    )
    X, y = np.asarray(seqs, dtype='float32'), np.asarray(labels)

    # Primero separa test
    X_rem, X_test, y_rem, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    # Luego separa val desde el resto
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_rem, y_rem, test_size=val_ratio,
        random_state=42, stratify=y_rem
    )

    # Guarda en disco
    os.makedirs(DATA_PATH, exist_ok=True)
    with open(os.path.join(DATA_PATH, "train_val_test.pkl"), "wb") as f:
        pickle.dump(
            dict(X_train=X_train, y_train=y_train,
                 X_val=X_val, y_val=y_val,
                 X_test=X_test, y_test=y_test,
                 word_ids=word_ids),
            f
        )
    print("Datasets guardados en data/train_val_test.pkl")

if __name__ == "__main__":
    main()
