# run_pipeline.py

import cv2
from ultralytics import YOLO

# Ruta al modelo entrenado
MODEL_PATH = "src/runs/detect/train/weights/best.pt"

def detectar_con_camara(model_path):
    # Cargar el modelo YOLOv11 (Ultralytics)
    model = YOLO(model_path)

    # Abrir cámara (0 = webcam por defecto)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("No se pudo abrir la cámara.")
        return

    print("Cámara activa. Presiona 'q' para salir.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo capturar el frame.")
            break

        # Realizar detección
        results = model(frame, conf=0.1)

        # Obtener imagen con anotaciones
        annotated_frame = results[0].plot()

        # Mostrar el frame con resultados
        cv2.imshow("Detección en tiempo real", annotated_frame)

        # Salir con la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    detectar_con_camara(MODEL_PATH)

if __name__ == "__main__":
    main()