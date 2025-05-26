import cv2
import numpy as np

def order_points(pts):
    # Ordena los puntos: top-left, top-right, bottom-right, bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def select_best_quadrilateral(image, contours, img_area, img_w, img_h):
    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < img_area * 0.05 or area > img_area * 0.95:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            squareness = min(aspect_ratio, 1/aspect_ratio)
            cx, cy = x + w/2, y + h/2
            center_dist = np.sqrt((cx - img_w/2)**2 + (cy - img_h/2)**2)
            center_score = 1 - center_dist / (0.5 * np.sqrt(img_w**2 + img_h**2))
            margin = min(x, y, img_w - (x + w), img_h - (y + h))
            score = area * squareness * center_score * (margin+1)
            candidates.append((score, approx))
    if not candidates:
        return None
    # Ordena por score descendente
    candidates.sort(reverse=True, key=lambda x: x[0])
    # Visualiza los candidatos y permite elegir con el teclado
    for idx, (score, approx) in enumerate(candidates):
        temp_img = image.copy()
        cv2.drawContours(temp_img, [approx], -1, (255,0,0), 3)
        cv2.putText(temp_img, f"Candidato {idx+1}/{len(candidates)} (Score: {int(score)})", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        
        key = cv2.waitKey(0)
        if key == 13 or key == 10:  # ENTER
            cv2.destroyAllWindows()
            return order_points(approx.reshape(4,2))
    cv2.destroyAllWindows()
    # Si no se selecciona ninguno, devuelve el mejor por score
    return order_points(candidates[0][1].reshape(4,2))

def process_chessboard(image, pattern_size=(7, 7), cell_size=60):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY,11,2)
    thresh = cv2.bitwise_not(thresh)
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    img_area = image.shape[0] * image.shape[1]
    img_h, img_w = image.shape[:2]

    # Selección robusta del cuadrilátero
    pts_src = select_best_quadrilateral(image, contours, img_area, img_w, img_h)
    if pts_src is None:
        raise ValueError("No se pudo detectar el borde interior del tablero.")

    out_size = cell_size * 8
    pts_dst = np.float32([
        [0, 0],
        [out_size-1, 0],
        [out_size-1, out_size-1],
        [0, out_size-1]
    ])
    M = cv2.getPerspectiveTransform(pts_src, pts_dst)
    top_down = cv2.warpPerspective(image, M, (out_size, out_size))

    # Detección de bordes y resaltado
    edges = cv2.Canny(cv2.cvtColor(top_down, cv2.COLOR_BGR2GRAY), 100, 200)
    top_down[edges > 0] = [0, 0, 255]

    # Matriz de coordenadas de casillas
    squares = np.zeros((8,8,4), dtype=int)  # (x1, y1, x2, y2) para cada casilla
    for i in range(8):
        for j in range(8):
            x1, y1 = i*cell_size, j*cell_size
            x2, y2 = (i+1)*cell_size, (j+1)*cell_size
            squares[i, j] = [x1, y1, x2, y2]
            cv2.rectangle(top_down, (x1, y1), (x2, y2), (0,255,0), 1)

    # Coordenadas de los bordes externos (en la imagen transformada)
    board_corners = pts_dst.astype(int)

    return top_down, squares, board_corners, M

def main():
    empty_img_path = 'ChessTracker_DavidBuitrago_CarlosGalan/images/empty.jpg'
    image_empty = cv2.imread(empty_img_path)
    if image_empty is None:
        print(f"Error cargando la imagen: {empty_img_path}")
        return

    processed_image, squares, board_corners, M = process_chessboard(image_empty)
    out_size = processed_image.shape[0]

    # Mostrar el resultado
    cv2.imshow("Tablero procesado", processed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#if __name__ == "__main__":  
   # main()