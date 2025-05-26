imports os, zipfile, requests, yaml, shutil, random
from pathlib import Path
from tqdm import tqdm
import torch
from ultralytics import YOLO
from pathlib import Path
import yaml
import random as rn
import shutil
import os
import re

#Enlazamos las imagenes con los labels

folder = "data/images"
folder_labels = "data/labels"

image_path = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith((".jpeg"))]
label_path = [os.path.join(folder_labels, f) for f in os.listdir(folder_labels) if f.endswith((".txt"))]
dataset = []

for path_i in image_path:
  index_i= re.findall(r'\d+', path_i)

  for path_l in label_path:
    index_l= re.findall(r'\d+', path_l)
    if index_i == index_l:
      dataset.append([path_i, path_l])
img_data =[]
label_data = []

for data in dataset:

  img_data.append(data[0])
  label_data.append(data[1])

#Creamos nuestros conjuntos de entrenamiento y validacion
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(img_data, label_data, test_size=0.2, random_state=42)

#Separacion en carpetas de nuestros datos de entrenamiento y validacion

X_train = [ Path(dir)  for dir in X_train]
y_train = [ Path(dir)  for dir in y_train]
X_val = [ Path(dir)  for dir in X_val]
y_val = [ Path(dir)  for dir in y_val]


destino_X_train = Path("/content/drive/MyDrive/dataset_YOLO/dataset/images/train")
destino_y_train = Path("/content/drive/MyDrive/dataset_YOLO/dataset/labels/train")
destino_X_val = Path("/content/drive/MyDrive/dataset_YOLO/dataset/images/val")
destino_y_val = Path("/content/drive/MyDrive/dataset_YOLO/dataset/labels/val")

destinos = [destino_X_train, destino_y_train, destino_X_val, destino_y_val]


for destino in destinos:
  if destino.exists():
      shutil.rmtree(destino)
  destino.mkdir(parents=True)


for (origen_x, origen_y) in zip(X_train, y_train):

    shutil.copy(origen_x, destino_X_train)
    shutil.copy(origen_y, destino_y_train)


for (origen_x, origen_y) in zip(X_val, y_val):

    shutil.copy(origen_x, destino_X_val)
    shutil.copy(origen_y, destino_y_val)

#Variables globales para el entrenamiento y analisis de precision

DATA_DIR = "/content/drive/MyDrive/dataset_YOLO/dataset"
YAML_OUT = "/content/drive/MyDrive/dataset_YOLO/dataset/custom.yaml"
MODEL_WEIGHTS = "yolov8n.pt"
EPOCHS = 300
IMG_SIZE = 640
FREEZE_LAYERS = 10

#Funcion para crear archivo yaml

def create_yaml(data_root: Path, yaml_path: Path):
    cfg = {
        "path": str(data_root),
        "train": "images/train",
        "val": "images/val",
        "test": "",
        "nc": 1,
        "names": {
            0: "silla_ocupada"
        }
    }
    with open(yaml_path, "w") as f:
        yaml.safe_dump(cfg, f)
    print(f"YAML creado en {yaml_path}")

#Funcion que realiza el Fine tuning de el modelo con los datos proporcionados

def fine_tune(yaml_path: Path, freeze_depth: int):
    print("Cargando modelo…")
    model = YOLO(MODEL_WEIGHTS)

    print(f"Iniciando fine‑tuning {EPOCHS} épocas, freeze={freeze_depth}")
    model.train(
        data=str(yaml_path),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=-1,
        freeze=freeze_depth,
        device='gpu',
        lr0=0.0005,
        patience=300,
        verbose=True,
    )
    return model

#Funcion de clasificacion binaria de una imagen a partir de un modelo que ha pasado por un proceso de fine-tunning

def clasificacion(model: YOLO, img):
    res = model(img)
    boxes = res[0].boxes
    ocupado = any([model.names[int(cls)] == "silla_ocupada" for cls in boxes.cls])
    return "clasificacion:OCUPADO" if ocupado else "clasificacion:DESOCUPADO"

#Funcion que analiza el modelo con los datos de validacion

def validate_and_infer(model: YOLO, yaml_path: Path):
    print("Validación…")
    metrics = model.val(data=str(yaml_path))
    print(metrics)


    val_imgs = list((Path(DATA_DIR) / "images" / "val").glob("*.jpeg"))

    for img in val_imgs:
        print(f"Inferencia en {img.name}")
        res = model(img)
        res[0].show()
        print(clasificacion(model, img))

torch.manual_seed(0)

create_yaml(DATA_DIR, YAML_OUT)

#Entrenamiendo del modelo
#Por favor volver a entrenar el modelo, esto debido a que ya sobrepasamos las limitaciones de GPU que nos proporciona Colab

yolo_model = fine_tune(YAML_OUT, FREEZE_LAYERS)

"""El modelo ha sido entrenado con learning rates=0.005,0.01 y epocas=50,100,200,300. Los mejores resultaron se obtuvieron en cantidades de epocas mas grandes y learning rate mas pequenos. las mediciones de box_loss y cls_loos fueron menores que el 0.4. lamentablementa por la poca cantidad de datos es posible que no haya una precision muy alta"""

validate_and_infer(yolo_model, YAML_OUT)

# modelo para segmentar imagenes
model_seg = YOLO("yolov8n-seg.pt")

#Segmentacion de imagenes obtenidas

val_imgs = list((Path(DATA_DIR) / "images" / "val").glob("*.jpeg"))

for img in val_imgs:
  print(f"Inferencia en {img.name}")
  res = model(img)
  res_seg=model_seg(res[0])
  res_seg[0].show()