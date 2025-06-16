import random

def get_random_move(board, current_player):
    legal_moves = []
    for i in range(len(board)):
        for j in range(len(board[0])):
            count, owner = board[i][j]
            if owner is None or owner == current_player:
                legal_moves.append((i, j))
    if legal_moves:
        return random.choice(legal_moves)
    return None
