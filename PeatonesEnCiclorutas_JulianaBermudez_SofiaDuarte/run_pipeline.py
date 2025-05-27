import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

def check_person_in_bike_lane(img_path, bike_lane_model, person_model,
                               bike_lane_class_id=0, person_class_id=0, bicycle_class_id=1,
                               conf=0.1):
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise ValueError(f"Could not read image at {img_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Segmentación de ciclorutas
    bike_result = bike_lane_model(img_rgb, conf=conf)[0]

    # Detección de personas y bicicletas
    detection_result = person_model(img_rgb)[0]

    if bike_result.masks is None or bike_result.boxes is None:
        print("No bike lanes detected.")
        return False

    masks = bike_result.masks.data.cpu().numpy()
    classes = bike_result.boxes.cls.cpu().numpy()
    mask_shape = img_rgb.shape[:2]

    # Combinar todas las máscaras de cicloruta
    bike_lane_mask = np.zeros(mask_shape, dtype=np.uint8)
    for i, cls_id in enumerate(classes):
        if int(cls_id) == bike_lane_class_id:
            mask = (masks[i] > 0.5).astype(np.uint8)
            resized_mask = cv2.resize(mask, (mask_shape[1], mask_shape[0]), interpolation=cv2.INTER_NEAREST)
            bike_lane_mask = np.maximum(bike_lane_mask, resized_mask)

    # Extraer cajas de personas y bicicletas
    boxes = detection_result.boxes.xyxy.cpu().numpy().astype(int)
    labels = detection_result.boxes.cls.cpu().numpy().astype(int)

    person_boxes = [boxes[i] for i in range(len(labels)) if labels[i] == person_class_id]
    bicycle_boxes = [boxes[i] for i in range(len(labels)) if labels[i] == bicycle_class_id]

    # Filtrar personas que NO estén sobre bicicletas (intersección IOU)
    valid_person_boxes = []
    for p_box in person_boxes:
        discard = False
        for b_box in bicycle_boxes:
            # Calcular intersección entre la persona y la bicicleta
            xA = max(p_box[0], b_box[0])
            yA = max(p_box[1], b_box[1])
            xB = min(p_box[2], b_box[2])
            yB = min(p_box[3], b_box[3])
            inter_area = max(0, xB - xA) * max(0, yB - yA)

            p_area = (p_box[2] - p_box[0]) * (p_box[3] - p_box[1])
            if inter_area / p_area > 0.2:  # umbral ajustable
                discard = True
                break
        if not discard:
            valid_person_boxes.append(p_box)

    detected = False
    # Verificar si las personas válidas pisan la cicloruta
    for box in valid_person_boxes:
        x1, y1, x2, y2 = box
        # Solo tomamos la parte inferior (pies)
        feet_box = (x1, y2 - int(0.2 * (y2 - y1)), x2, y2)
        person_mask = np.zeros_like(bike_lane_mask, dtype=np.uint8)
        person_mask[feet_box[1]:feet_box[3], feet_box[0]:feet_box[2]] = 1

        overlap = np.logical_and(bike_lane_mask, person_mask)
        if np.any(overlap):
            detected = True
            break

    # Visualización
    overlay = img_rgb.copy()
    overlay[bike_lane_mask == 1] = [255, 0, 0]  # rojo para la cicloruta
    for box in valid_person_boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)  # verde para personas

    plt.figure(figsize=(8, 8))
    plt.title("Detección de personas y ciclorutas")
    plt.imshow(overlay)
    plt.axis("off")
    plt.show()

    if detected:
        print("Person walking in bike lane detected.")
        return True
    else:
        print("No person in bike lane.")
        return False


# Se cargan los modelos

bike_lane_model = YOLO("src/yolo11n-seg-finetuned2/weights/best.pt")
person_model = YOLO("yolo11n.pt")

# Se pone la dirección de la imange
image_path = "data/images/prueba/cicloruta25.jpg"

# Se llama la función
result = check_person_in_bike_lane(image_path, bike_lane_model, person_model)
print("Result:", result)
