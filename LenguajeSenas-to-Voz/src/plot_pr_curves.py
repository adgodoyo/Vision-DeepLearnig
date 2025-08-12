import pickle, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.metrics import precision_recall_curve, average_precision_score
from constants import *

sns.set(style="whitegrid")

# ── 1. cargar modelo y test set ──────────────────────────────────────
model = load_model(MODEL_PATH)

with open(os.path.join(DATA_PATH, "train_val_test.pkl"), "rb") as f:
    d = pickle.load(f)

X_test, y_true, word_ids = d["X_test"], d["y_test"], d["word_ids"]
n_classes = len(word_ids)

# ── 2. inferencia probabilística ────────────────────────────────────
y_score = model.predict(X_test, verbose=0)        # shape (N, 10)
y_bin   = np.eye(n_classes)[y_true]               # one-hot ground-truth

# ── 3. curva PR por clase ───────────────────────────────────────────
plt.figure(figsize=(7, 5))
for i, name in enumerate(word_ids):
    p, r, _ = precision_recall_curve(y_bin[:, i], y_score[:, i])
    ap      = average_precision_score(y_bin[:, i], y_score[:, i])
    plt.plot(r, p, label=f"{name}  (AP={ap:.2f})")

plt.xlabel("Recall");  plt.ylabel("Precision")
plt.title("Curvas Precision–Recall (set TEST)")
plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
plt.tight_layout();  plt.show()
