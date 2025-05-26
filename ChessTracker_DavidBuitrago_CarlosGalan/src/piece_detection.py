from ultralytics import YOLO
import cv2
import numpy as np
def piece_detection(image_path, model_path, conf=0.5):
    """
    Detecta piezas de ajedrez en una imagen usando un modelo YOLO adaptado a nuestro dataset.
    Muestra la imagen con las predicciones y guarda el resultado.
    """
    # Cargar el modelo entrenado
    model = YOLO(model_path) 

    # Realizar la predicción
    results = model.predict(source=image_path, save=True, conf=conf)  # conf = umbral de confianza
    return results
    
def asignar_piezas_a_casillas_transform(detections, squares, class_names, M):
    """
    Asigna piezas detectadas en la imagen original a las casillas del tablero wrappeado,
    usando un punto más cerca de la base (parte inferior) del bounding box.
    """
    board_state = [["" for _ in range(8)] for _ in range(8)]
    for det in detections:
        x1, y1, x2, y2 = map(int, det.xyxy[0])
        # Tomar un punto al 85% de la altura (más cerca de la base)
        base_y = y2
        cx = (x1 + x2) // 2
        cy = base_y
        # Transformar el punto a la imagen wrappeada
        src_pt = np.array([[[cx, cy]]], dtype=np.float32)
        dst_pt = cv2.perspectiveTransform(src_pt, M)
        cx_warp, cy_warp = int(dst_pt[0][0][0]), int(dst_pt[0][0][1])
        cls = int(det.cls[0])
        label = class_names[cls] if 0 <= cls < len(class_names) else str(cls)
        for i in range(8):
            for j in range(8):
                sx1, sy1, sx2, sy2 = squares[i, j]
                if sx1 <= cx_warp < sx2 and sy1 <= cy_warp < sy2:
                    board_state[i][j] = label
                    break
    return board_state


#if __name__ == "__main__":
    #image_path = 'ChessTracker_DavidBuitrago_CarlosGalan/images/prev.jpg'  # Cambia a tu imagen
    #model_path = 'ChessTracker_DavidBuitrago_CarlosGalan/models/best.pt'  # Cambia a tu modelo
    #piece_detection(image_path, model_path)

