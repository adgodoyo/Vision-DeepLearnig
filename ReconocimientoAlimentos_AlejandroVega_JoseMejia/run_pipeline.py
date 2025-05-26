# run_pipeline.py

import cv2
from ultralytics import YOLO
from torchvision import models
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

# Ruta a imagen de nevera a revisar

nevera_path = 'data/nevera_prueba.jpg'
# nevera_path = 'data/nevera_casivacia.jpg'
# nevera_path = 'data/nevera_mediollena.jpg'

features_empty = np.load('src/runs/embedding_vacio.npy')
features_full = np.load('src/runs/embedding_lleno.npy')

# Modelo para embeddings
model = models.resnet50(pretrained=False)
model = torch.nn.Sequential(*(list(model.children())[:-1]))


model.load_state_dict(torch.load('src/runs/resnet50_feature_extractor.pth'))
model.eval()

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def histogram_eq(image):
    # Convert any PIL Image to OpenCV format
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    # Equalize the Y (luminance) channel
    img[:,:,0] = cv2.equalizeHist(img[:,:,0])
    # Convert back to RGB
    img = cv2.cvtColor(img, cv2.COLOR_YCrCb2RGB)
    return Image.fromarray(img)

def get_features(img_path):
    img = Image.open(img_path).convert('RGB')
    img = histogram_eq(img)  # Ecualización
    x = transform(img).unsqueeze(0)
    with torch.no_grad():
        feats = model(x).squeeze().numpy().flatten()
    feats = feats / np.linalg.norm(feats)
    return feats

features_test = get_features(nevera_path)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

sim_to_empty = cosine_similarity(features_test, features_empty)
sim_to_full = cosine_similarity(features_test, features_full)

# Se calculan distancias a lleno y vacío
distance_to_empty = 1 - sim_to_empty
distance_to_full = 1 - sim_to_full

# Se normaliza el score de que tal llena está la nevera
fullness_score = distance_to_empty / (distance_to_empty + distance_to_full)

fullness_percent = int(fullness_score * 100)

# Ruta al modelo entrenado
MODEL_PATH = "src/runs/detect/train/weights/best.pt"
yolo = YOLO(MODEL_PATH)

class_list = ['Arepa', 'Arequipe', 'Atun', 'BonYurt', 'Cerveza', 'Coca Cola', 'Jugo de Mandarina',
              'Jugo de Naranja', 'Leche', 'Mayonesa', 'Mermelada', 'Mostaza', 'Queso Mozzarella',
              'Queso Parmesano', 'Salsa de Tomate', 'Saltin Noel', 'Suero', 'Yogurt Griego']

nevera = cv2.imread(nevera_path)

nevera = cv2.resize(nevera, (480,640), interpolation=cv2.INTER_AREA)

results = yolo(nevera, conf=0.05, verbose = False)

nevera = results[0].plot()

detected_class_indices = results[0].boxes.cls.int().tolist()

detected_classes = set([yolo.model.names[idx] for idx in detected_class_indices])

remaining_classes = [cls for cls in class_list if cls not in detected_classes]

if detected_classes:
    print(f"Hay {', '.join(detected_classes)} en la nevera\n")
else:
    print('No hay productos de tu lista de compras en la nevera\n')
    
print(f"Necesitas comprar {', '.join(remaining_classes)}.\n\n")

cv2.imshow(f"Nevera {fullness_percent}% llena", nevera)

print(f"La nevera está un {fullness_percent:.2f}% llena")

# video_path = 'src/IMG_5900_rotated.mp4' # Otro video de prueba
# rotar = False # Ajustar para rotacion del video

video_path = 'data/video_prueba.mp4' 
rotar = True # Ajustar para rotacion del video


def detectar_productos(model_path, remaining_classes=remaining_classes, video_path=video_path, usar_video=False, rotar=rotar):
    model = YOLO(model_path)

    cap = cv2.VideoCapture(0)
    camera_opened = cap.isOpened()
    
    if not camera_opened or usar_video:
        
        video = True

        print(f"Usando archivo de video: {video_path}\n\n")
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print("No se pudo abrir el archivo de video.")
            return
    else:
        video = False
        print("Cámara activa. Presiona 'q' para salir.\n\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo capturar el frame. Fin del video o error.")
            break
        
        if video and rotar:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            frame = cv2.resize(frame, (480,640), interpolation=cv2.INTER_AREA)
        
        # Realizar detección
        results = model(frame, conf=0.5, verbose=False)
        
        detected_class_indices = results[0].boxes.cls.int().tolist()
        detected_classes = set([model.names[idx] for idx in detected_class_indices])
        
        for producto in detected_classes:
            if producto in remaining_classes:
                print(f'Hay {producto} en el supermercado.')
                remaining_classes.remove(producto)
                print(f"Necesitas comprar {', '.join(remaining_classes)}.\n")

        # Obtener imagen con anotaciones
        annotated_frame = results[0].plot()

        # Mostrar el frame con resultados
        cv2.imshow("Detección", annotated_frame)

        # Salir con la tecla 'q' o si el video terminó
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
def main():
    detectar_productos(MODEL_PATH, usar_video=False)

if __name__ == "__main__":
    main()