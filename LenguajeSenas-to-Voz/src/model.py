from tensorflow.keras import layers, models
from constants import LENGTH_KEYPOINTS

def tcn_block(x, n_filters, dilation, dropout=0.3):
    prev = x
    for _ in range(2):
        x = layers.Conv1D(n_filters, 3, padding="causal",
                          dilation_rate=dilation, activation="relu")(x)
        x = layers.Dropout(dropout)(x)
    if prev.shape[-1] != n_filters:
        prev = layers.Conv1D(n_filters, 1, padding="same")(prev)
    x = layers.Add()([x, prev])
    return layers.LayerNormalization()(x)

def get_model(max_len, n_classes):
    inp = layers.Input((max_len, LENGTH_KEYPOINTS))
    x = inp
    # receptive field 31 frames
    for d in (1,2,4,8,16):
        x = tcn_block(x, 256, dilation=d)
    # simple attention
    att = layers.Dense(1, activation='tanh')(x)
    att = layers.Softmax(axis=1)(att)
    x = layers.Multiply()([x, att])
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(n_classes, activation='softmax')(x)
    model = models.Model(inp, out)
    model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
    return model
