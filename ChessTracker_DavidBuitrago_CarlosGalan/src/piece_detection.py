import cv2
import numpy as np
from ultralytics import YOLO

# Cargar modelo YOLO preentrenado (ajusta la ruta a tus pesos)
model = YOLO('models/yolo_chess.pt')  # e.g., best.pt descargado o entrenado


def recognize_pieces_yolo(board_img, conf_threshold=0.5, img_size=640):
    """
    Reconoce piezas en una imagen de tablero warpeado usando YOLO.
    Devuelve un estado de 8x8 con etiquetas como 'wP', 'bK', etc., o '.' si está vacía.

    Args:
        board_img (np.array): imagen BGR top-down del tablero (square_size*8).
        conf_threshold (float): umbral de confianza para filtrar detecciones.
        img_size (int): tamaño de reescalado para la inferencia de YOLO.

    Returns:
        list[list[str]]: matriz 8x8 con etiquetas de piezas o '.'.
    """
    # Ejecutar inferencia
    result = model(board_img, imgsz=img_size, conf=conf_threshold)[0]

    # Inicializar tablero vacío
    board_state = [['.' for _ in range(8)] for _ in range(8)]
    h, w = board_img.shape[:2]
    square_size = h // 8

    # Procesar cada detección
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        cls_id = int(box.cls[0].cpu().numpy())
        label = model.names[cls_id]

        # Centro de la caja para determinar casilla
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        row = int(cy // square_size)
        col = int(cx // square_size)

        # Asignar etiqueta en la casilla correspondiente
        if 0 <= row < 8 and 0 <= col < 8:
            board_state[row][col] = label

    return board_state