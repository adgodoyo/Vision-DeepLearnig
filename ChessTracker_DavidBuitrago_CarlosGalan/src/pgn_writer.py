def move_to_pgn(piece, move_from, move_to, captured_piece=None, castling=None):
    files = 'abcdefgh'
    ranks = '87654321'
    from_i, from_j = move_from
    to_i, to_j = move_to

    piece_map = {
        'king': 'K', 'queen': 'Q', 'rook': 'R', 'bishop': 'B', 'knight': 'N', 'pawn': ''
    }
    if '-' in piece:
        _, piece_type = piece.split('-', 1)
    else:
        piece_type = piece
    pgn_piece = piece_map.get(piece_type.lower(), '')

    from_square = files[from_j] + ranks[from_i]
    to_square = files[to_j] + ranks[to_i]

    # Enroque
    if castling:
        return castling

    # Captura
    if captured_piece and captured_piece != "":
        if pgn_piece == "":  # Peón captura
            return f"{files[from_j]}x{to_square}"
        else:
            return f"{pgn_piece}x{to_square}"

    # Movimiento normal
    return f"{pgn_piece}{to_square}"