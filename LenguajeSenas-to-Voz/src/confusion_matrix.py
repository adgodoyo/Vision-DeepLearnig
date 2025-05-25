import pickle, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from tensorflow.keras.models import load_model
from constants import *

def evaluate(model_path: str):
    # 1. Carga modelo y test-set
    model = load_model(model_path)
    with open(os.path.join(DATA_PATH, "train_val_test.pkl"), "rb") as f:
        data = pickle.load(f)

    X_test = data["X_test"]
    y_true = data["y_test"]
    word_ids = data["word_ids"]

    # 2. Predicción
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

    # 3. Métricas detalladas
    print("\n Classification report:")
    print(classification_report(y_true, y_pred, target_names=word_ids))

    acc = accuracy_score(y_true, y_pred)
    print(f"Global accuracy (TEST): {acc:.4f}")

    # 4. Matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=word_ids, yticklabels=word_ids)
    plt.xlabel("Predicho"); plt.ylabel("Real"); plt.title("Matriz de Confusión (TEST)")
    plt.tight_layout(); plt.show()

if __name__ == "__main__":
    evaluate(MODEL_PATH)
