def boardstate_to_fen(board_state):
    """
    Convierte una matriz 8x8 de piezas (o '1' para vacía) a una fila FEN.
    """
    fen_rows = []
    for row in board_state:
        fen_row = ''
        empty_count = 0
        for piece in row:
            if piece == "" or piece == "empty":
                empty_count += 1
            else:
                if empty_count > 0:
                    fen_row += str(empty_count)
                    empty_count = 0
                # Mapeo de pieza a letra FEN (ajusta según tus nombres de clase)
                piece_map = {
                    'white-king': 'K', 'white-queen': 'Q', 'white-rook': 'R', 'white-bishop': 'B', 'white-knight': 'N', 'white-pawn': 'P',
                    'black-king': 'k', 'black-queen': 'q', 'black-rook': 'r', 'black-bishop': 'b', 'black-knight': 'n', 'black-pawn': 'p',
                    'bishop': 'B', 'knight': 'N', 'rook': 'R', 'queen': 'Q', 'king': 'K', 'pawn': 'P',
                    'black-bishop': 'b', 'black-knight': 'n', 'black-rook': 'r', 'black-queen': 'q', 'black-king': 'k', 'black-pawn': 'p',
                    'white-bishop': 'B', 'white-knight': 'N', 'white-rook': 'R', 'white-queen': 'Q', 'white-king': 'K', 'white-pawn': 'P'
                }
                fen_row += piece_map.get(piece, '1')
        if empty_count > 0:
            fen_row += str(empty_count)
        fen_rows.append(fen_row)
    return '/'.join(fen_rows)
