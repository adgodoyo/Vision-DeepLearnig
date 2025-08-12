import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import tensorflow as tf

# Configuraciones
plt.rcParams["figure.dpi"] = 110
plt.style.use("ggplot")
tf.random.set_seed(42)
np.random.seed(42)

def pipeline(image):
    labels = ["Adios", "Gracias", "No", "Por favor", "Si"]
    h, w, _ = image.shape
    loc = tf.keras.models.load_model("src/results/localizer.keras")
    seg = tf.keras.models.load_model("src/results/segmenter.keras")
    clf = tf.keras.models.load_model("src/results/classifier.keras")
    
    # Localization
    img1 = tf.image.resize(image, (128, 128))
    img1_prep = tf.keras.applications.mobilenet_v2.preprocess_input(img1[tf.newaxis, ...])
    p1 = loc.predict(img1_prep)[0]
    x1, y1 = int(p1[0]*w), int(p1[1]*h)
    x2, y2 = int((p1[0] + p1[2])*w), int((p1[1] + p1[3])*h)
    cv.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)

    # Segmentation
    image2 = image[x1:x2, y1:y2, :]
    img2 = tf.image.resize(image2, (128, 128))
    img2_prep = tf.keras.applications.mobilenet_v2.preprocess_input(img2[tf.newaxis, ...])
    p2 = seg.predict(img2_prep)[0]
    p2 = tf.math.argmax(p2, axis=-1)
    p2 = np.expand_dims(p2, axis=-1)

    # Classification
    img3 = np.where(p2 == 1, img2, 0)
    img3_prep = tf.keras.applications.mobilenet_v2.preprocess_input(img3[tf.newaxis, ...])
    p3 = np.argmax(clf.predict(img3_prep))
    cv.putText(image, labels[p3], (110, 100), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 6, cv.LINE_AA)
    return image

image = cv.imread("data/localization/si/si_0.jpg")
cv.imshow("image", pipeline(image.copy())) # Si se corre en Google Colab, es mejor con cv_show