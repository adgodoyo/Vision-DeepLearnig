import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
from roboflow import Roboflow
from github import Github

# ==============================
ROBOFLOW_API_KEY = "Ouks0aNv3B8r7IcgQDRI"
GITHUB_TOKEN = "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"
GITHUB_REPO = 'Mapa-rampas'
GITHUB_PAGE_URL = 'https://laurar287.github.io/Mapa-rampas/'

# Directorios base
BASE_DIR = os.path.dirname(__file__)
TEST_DIR = os.path.join(BASE_DIR, 'data', 'Test')
MODELS_DIR = os.path.join(BASE_DIR, 'data', 'Modelos')

# Configurar Roboflow
rf = Roboflow(api_key=ROBOFLOW_API_KEY)

# Detección
project_det = rf.workspace("vision-computacional-u02en").project("ramps-detection-n1ecd")
version_det = project_det.version(1)
version_det.download("yolov11")

# Segmentación
project_seg = rf.workspace("vision-computacional-u02en").project("ramps-segmentation")
version_seg = project_seg.version(1)
version_seg.download("yolov11")

# Entrenar modelos
detection_model = YOLO('yolo11n.pt')
detection_model.train(
    data=os.path.join(BASE_DIR, 'Ramps-detection-1', 'data.yaml'),
    epochs=150,
    imgsz=640,
    batch=16,
    name='rampa_detection_model',
    augment=True,
    patience=20
)
detection_model.predict(os.path.join(BASE_DIR, 'Ramps-detection-1', 'test', 'images', '*'), imgsz=640, save=True, conf=0.5)

segmentation_model = YOLO('yolo11n-seg.pt')
segmentation_model.train(
    data=os.path.join(BASE_DIR, 'Ramps-segmentation-1', 'data.yaml'),
    epochs=50,
    imgsz=640,
    batch=16,
    name='rampa_segmentation_model'
)
segmentation_model.predict(os.path.join(BASE_DIR, 'Ramps-segmentation-1', 'test', 'images', '*'), imgsz=640, save=True, conf=0.5)

# Inferencia final
rampas_df = pd.read_csv(os.path.join(TEST_DIR, 'RampsGeoreference.csv'))
detection_model = YOLO(os.path.join(MODELS_DIR, 'best_detection.pt'))
segmentation_model = YOLO(os.path.join(MODELS_DIR, 'best_segmentation.pt'))

image_paths = [os.path.join(TEST_DIR, f) for f in os.listdir(TEST_DIR) if f.endswith('.png')]

# Conversión de píxel a coordenadas geográficas
def pixel_to_geo(x_pixel, y_pixel, width, height, lat, lon):
    lat_offset = (y_pixel - height / 2) / height * 0.001
    lon_offset = (x_pixel - width / 2) / width * 0.001
    return lat + lat_offset, lon + lon_offset

# DataFrame resultados
results_df = pd.DataFrame(columns=['image_id', 'latitude', 'longitude', 'rampa_id', 'rampa_lat', 'rampa_lon'])

# Procesamiento de cada imagen
for path in image_paths:
    image_id = os.path.splitext(os.path.basename(path))[0]
    image = cv2.imread(path)
    if image is None:
        continue
    h, w, _ = image.shape
    row = rampas_df[rampas_df['image_id'] == image_id]
    if row.empty:
        continue
    lat, lon = row['latitude'].values[0], row['longitude'].values[0]
    results = detection_model.predict(image, conf=0.5)
    if results and results[0].boxes:
        for i, box in enumerate(results[0].boxes.xywh):
            x, y, bw, bh = box[:4].tolist()
            r_lat, r_lon = pixel_to_geo(x, y, w, h, lat, lon)
            cv2.rectangle(image, (int(x - bw / 2), int(y - bh / 2)), (int(x + bw / 2), int(y + bh / 2)), (0, 255, 0), 2)
            seg_res = segmentation_model.predict(image, conf=0.3)
            if seg_res and seg_res[0].masks is not None:
                for seg_mask in seg_res[0].masks.data:
                    mask = seg_mask.cpu().numpy()
                    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                    mask_bin = (mask > 0.5).astype('uint8') * 255
                    mask_rgb = cv2.cvtColor(mask_bin, cv2.COLOR_GRAY2BGR)
                    image = cv2.addWeighted(image, 1, mask_rgb, 0.5, 0)
            results_df = pd.concat([results_df, pd.DataFrame([{
                'image_id': image_id,
                'latitude': lat,
                'longitude': lon,
                'rampa_id': i + 1,
                'rampa_lat': r_lat,
                'rampa_lon': r_lon
            }])], ignore_index=True)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(f"{image_id}")
    plt.axis('off')
    plt.show()

# Guardar CSV
csv_output_path = os.path.join(BASE_DIR, 'rampas_con_coordenadas.csv')
results_df.to_csv(csv_output_path, index=False)
print("CSV guardado exitosamente.")

# Subida a GitHub y mostrar URL
if GITHUB_TOKEN:
    g = Github(GITHUB_TOKEN)
    repo = g.get_user().get_repo(GITHUB_REPO)
    with open(csv_output_path, 'r') as f:
        content = f.read()
    try:
        existing = repo.get_contents('rampas_con_coordenadas.csv')
        repo.update_file(existing.path, "Actualización CSV", content, existing.sha, branch="main")
        print("Archivo actualizado en GitHub.")
    except:
        repo.create_file('rampas_con_coordenadas.csv', "Creación CSV", content, branch="main")
        print("Archivo creado en GitHub.")
    print(f"✅ Puedes ver el resultado en: {GITHUB_PAGE_URL}")
