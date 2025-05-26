import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from matplotlib import pyplot as plt

# --- CONFIGURACIÓN ---
BASE_DIR = Path(__file__).resolve().parent
IMG_DIR = os.path.join(BASE_DIR, "data", "deteccion", "train", "images")

DETECTION_MODEL_PATH = os.path.join(BASE_DIR, "data", "modelodetec", "weights", "best.pt")
SEGMENTATION_MODEL_PATH = os.path.join(BASE_DIR, "data", "modeloseg", "weights", "best.pt")

OCCUPANCY_THRESHOLDS = {
    "Low": (0, 5),
    "Medium": (6, 12),
    "Full": (13, float("inf"))
}

def clasificar_por_area(pct):
    if pct <= 8:
        return "low"
    elif pct <= 18:
        return "medium"
    else:
        return "full"


# --- CARGA DE MODELOS ---
print("Cargando modelos...")
det_model = YOLO(DETECTION_MODEL_PATH)
seg_model = YOLO(SEGMENTATION_MODEL_PATH)

# --- FUNCIÓN DE CLASIFICACIÓN ---
def classify_occupancy(person_count):
    for label, (min_count, max_count) in OCCUPANCY_THRESHOLDS.items():
        if min_count <= person_count <= max_count:
            return label
    return "Unknown"

# --- PROCESAMIENTO DE IMÁGENES ---
output_dir = os.path.join(BASE_DIR, "output")
os.makedirs(output_dir, exist_ok=True)

for image_file in os.listdir(IMG_DIR):
    if not image_file.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    image_path = os.path.join(IMG_DIR, image_file)
    img = cv2.imread(image_path)

    # Detección
    results_det = det_model(img)
    boxes = results_det[0].boxes
    person_boxes = [box for box in boxes if int(box.cls) == 0] 

    person_count = len(person_boxes)
    occupancy_level = classify_occupancy(person_count)

    # Segmentación
    results_seg = seg_model(img)
    masks = results_seg[0].masks.data.cpu().numpy() if results_seg[0].masks else []
    total_mask_area = sum(np.sum(mask) for mask in masks)
    H, W = img.shape[:2]
    total_area = H * W
    pct_ocupado = (total_mask_area / total_area) * 100
    clase_area = clasificar_por_area(pct_ocupado)

    # Visualización
    annotated_img = results_det[0].plot()
    if len(masks) != 0:
        mask_img = np.zeros_like(img)
        for m in masks:
            mask = m.astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(mask_img, contours, -1, (0, 255, 0), 2)
        annotated_img = cv2.addWeighted(annotated_img, 1, mask_img, 0.5, 0)

    out_path = os.path.join(output_dir, f"result_{image_file}")
    cv2.imwrite(out_path, annotated_img)

    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title(f"Deteccion: {occupancy_level} ({person_count} personas). Segmentacion: {clase_area}")
    plt.show()
