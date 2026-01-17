import cv2
import numpy as np
from src.board_detection import process_chessboard
from src.piece_detection import piece_detection, asignar_piezas_a_casillas_transform
from src.fen_conversion import boardstate_to_fen
from src.move_detection import detectar_movimiento
from src.pgn_writer import move_to_pgn

def get_board_state(image_path, model_path, squares, class_names, M, conf=0.2):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error cargando la imagen: {image_path}")
        return None
    results = piece_detection(image, model_path, conf=conf)
    detections = results[0].boxes
    board_state = asignar_piezas_a_casillas_transform(detections, squares, class_names, M)
    return board_state

def main():
    # 1. Calibrar el tablero vacío
    empty_img_path = 'images\empty.JPG'
    image_empty = cv2.imread(empty_img_path)
    if image_empty is None:
        print(f"Error cargando la imagen: {empty_img_path}")
        return

    processed_image, squares, board_corners, M = process_chessboard(image_empty)
    out_size = processed_image.shape[0]

    # 2. Definir nombres de clases y modelo
    class_names = [
        'black-bishop','black-king','black-knight','black-pawn','black-queen','black-rook',
        'board','white-bishop','white-king','white-knight','white-pawn','white-queen','white-rook'
    ]
    model_path = r'models\best2.pt'

    def invertir_pos(pos):
        if pos is None:
            return None
        i, j = pos
        return (7 - i, j)

    pgn_moves = []
    fen_prev = None

    for i in range(1, 16):  # 1-14 como prev, 2-15 como curr
        prev_img_path = f'images/{i}.jpg'
        curr_img_path = f'images/{i+1}.jpg'

        prev_state = get_board_state(prev_img_path, model_path, squares, class_names, M)
        curr_state = get_board_state(curr_img_path, model_path, squares, class_names, M)

        if prev_state is None or curr_state is None:
            print(f"Error obteniendo los estados del tablero para {i} y {i+1}.")
            continue

        def rotar_90_izquierda(tablero):
            # Rota la matriz 90 grados a la izquierda (counterclockwise)
            return [list(fila) for fila in zip(*tablero)][::-1]
        
        def invertir_columnas(tablero):
            # Invierte solo las columnas de cada fila
            return [fila[::-1] for fila in tablero]

        # ...en tu bucle principal o donde generas el FEN...
        fen_prev = boardstate_to_fen(rotar_90_izquierda(invertir_columnas(prev_state)))
        fen_curr = boardstate_to_fen(rotar_90_izquierda(invertir_columnas(curr_state)))
        

        pieza, desde, hasta, captura, enroque = detectar_movimiento(prev_state, curr_state)
        desde_inv = invertir_pos(desde)
        hasta_inv = invertir_pos(hasta)
        if desde_inv is not None and hasta_inv is not None:
            pgn = move_to_pgn(pieza, desde_inv, hasta_inv, captured_piece=captura, castling=enroque)
            print(f"Movimiento {i}: {pgn}")
            print("FEN previo:", fen_prev)
            print("https://lichess.org/analysis/" + fen_prev)
            print("FEN actual:", fen_curr)
            print("https://lichess.org/analysis/" + fen_curr)
            pgn_moves.append((pgn, fen_prev, fen_curr))
        else:
            print(f"No se detectó movimiento entre {i} y {i+1}.")
            pgn_moves.append(("NoMove", fen_prev, fen_curr))

    # Guardar el PGN en un archivo
        # Guardar el PGN en un archivo compatible con Lichess desde posición personalizada
    initial_fen = pgn_moves[0][1] if pgn_moves and pgn_moves[0][1] else "startpos"
    with open("partida.pgn", "w", encoding="utf-8") as f:
        f.write('[Event "ChessTracker"]\n')
        f.write('[Site "Local"]\n')
        f.write('[Variant "From Position"]\n')
        f.write('[SetUp "1"]\n')
        f.write(f'[FEN "{initial_fen}"]\n')
        f.write('[Result "*"]\n\n')
        for idx, (move, fen_prev, fen_curr) in enumerate(pgn_moves):
            if move == "NoMove":
                continue
            if idx % 2 == 0:
                f.write(f"{(idx // 2) + 1}. ")
            # Escribe la jugada y la FEN previa como comentario
            f.write(f"{move} {{{fen_prev}}} ")
        f.write("*\n")

if __name__ == "__main__":
    main()