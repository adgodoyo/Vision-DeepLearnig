import pickle, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from tensorflow.keras.models import load_model, Model
from sklearn.manifold import TSNE          # si prefieres UMAP → pip install umap-learn
from constants import *

sns.set(style="white", rc={"figure.figsize":(6,5)})

# ── 1. modelo truncado hasta Dense(256) ─────────────────────────────
full_model = load_model(MODEL_PATH)
# penúltima capa (Dense 256) → cambia el índice si modificas la red
latent_layer = full_model.layers[-3].output
encoder      = Model(full_model.input, latent_layer)

# ── 2. cargar test set ──────────────────────────────────────────────
with open(os.path.join(DATA_PATH, "train_val_test.pkl"), "rb") as f:
    d = pickle.load(f)

X_test, y_test, word_ids = d["X_test"], d["y_test"], d["word_ids"]

# ── 3. obtener embeddings latentes ──────────────────────────────────
Z = encoder.predict(X_test, verbose=0)            # (N, 256)

# ── 4. reducir a 2-D (t-SNE) ────────────────────────────────────────
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
Z_2d = tsne.fit_transform(Z)

# ── 5. scatter plot coloreado por clase ─────────────────────────────
palette = sns.color_palette("hls", len(word_ids))
plt.figure()
for i, name in enumerate(word_ids):
    idx = y_test == i
    plt.scatter(Z_2d[idx,0], Z_2d[idx,1], s=25,
                color=palette[i], label=name, alpha=.8)

plt.title("t-SNE del espacio latente (TEST)")
plt.xticks([]); plt.yticks([])
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
plt.tight_layout(); plt.show()
