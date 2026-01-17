def detectar_movimiento(prev_state, curr_state):
    """
    Detecta el movimiento realizado entre dos estados del tablero (previo y actual).
    Devuelve (pieza, from_pos, to_pos, captured_piece, castling)
    """
    move_from = None
    move_to = None
    piece = None
    captured_piece = None
    castling = None

    # Buscar diferencias: casillas que cambian
    salidas = []
    llegadas = []
    for i in range(8):
        for j in range(8):
            prev = prev_state[i][j]
            curr = curr_state[i][j]
            if prev != "" and curr == "":
                salidas.append((i, j))
            if prev != curr and curr != "":
                llegadas.append((i, j))

    # Movimiento simple o captura (asume solo un movimiento por vez)
    if len(salidas) == 1 and len(llegadas) == 1:
        move_from = salidas[0]
        move_to = llegadas[0]
        piece = prev_state[move_from[0]][move_from[1]]
        # Captura si había una pieza en move_to en el estado previo
        captured_piece = prev_state[move_to[0]][move_to[1]] if prev_state[move_to[0]][move_to[1]] != "" else None

        # Detectar enroque (rey mueve dos columnas)
        if piece and 'king' in piece:
            if abs(move_from[1] - move_to[1]) == 2:
                castling = 'O-O' if move_to[1] > move_from[1] else 'O-O-O'

    return piece, move_from, move_to, captured_piece, castling