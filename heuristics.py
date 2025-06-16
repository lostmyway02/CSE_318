import random

# 2,4,6,9 favour Human

# If ai's orb count > opponent's orb count, then a positive score for ai
def heuristic_1(board, ai_player):
    score = 0
    for row in board:
        for count, owner in row:
            if owner == ai_player:
                score += count
            elif owner and owner != ai_player:
                score -= count
    return score                


# Self-sabotaging heuristic: ai tries to improve opponent's orb count
def heuristic_2(board, ai_player):
    score = 0
    for row in board:
        for count, owner in row:
            if owner == ai_player:
                score -= count
            elif owner and owner != ai_player:
                score += count
    return score    

# Corner-controlling heuristic
def heuristic_3(board, ai_player):
    rows = len(board)
    cols = len(board[0])
    score = 0

    corner_positions = [(0, 0), (0, cols - 1), (rows - 1, 0), (rows - 1, cols - 1)]

    for i in range(rows):
        for j in range(cols):
            count, owner = board[i][j]
            if owner == ai_player:
                score += count  # base score for owning orbs

                if (i, j) in corner_positions:
                    score += 2

            elif owner and owner != ai_player:
                score -= count 

    return score


# Self-sabotaging Corner-controlling heuristic
def heuristic_4(board, ai_player):
    rows = len(board)
    cols = len(board[0])
    score = 0

    corner_positions = [(0, 0), (0, cols - 1), (rows - 1, 0), (rows - 1, cols - 1)]

    for i in range(rows):
        for j in range(cols):
            count, owner = board[i][j]
            if owner == ai_player:
                score -= count  # base score for owning orbs

                if (i, j) in corner_positions:
                    score -= 2

            elif owner and owner != ai_player:
                score += count 

    return score

# Edge-controlling heuristic
def heuristic_5(board, ai_player):
    rows = len(board)
    cols = len(board[0])
    score = 0
    edge_cells = []
    
    # Edge consists of top, bottom, left, right cloumns excluding corners
    # Top & Bottom
    for j in range(1, cols - 1):
        edge_cells.append((0,j))
        edge_cells.append((rows - 1, j))
        
    # Left & Right
    for i in range(1, rows-1):
        edge_cells.append((i,0))
        edge_cells.append((i, cols -1))
        
    for i in range(rows):
        for j in range(cols):
            count, owner = board[i][j]
            if owner == ai_player:
                score += count  # base score for owning orbs

                if (i, j) in edge_cells:
                    score += 1

            elif owner and owner != ai_player:
                score -= count 

    return score

# Self-sabotaging Edge-controlling heuristic
def heuristic_6(board, ai_player):
    rows = len(board)
    cols = len(board[0])
    score = 0
    edge_cells = []
    
    # Edge consists of top, bottom, left, right cloumns excluding corners
    # Top & Bottom
    for j in range(1, cols - 1):
        edge_cells.append((0,j))
        edge_cells.append((rows - 1, j))
        
    # Left & Right
    for i in range(1, rows-1):
        edge_cells.append((i,0))
        edge_cells.append((i, cols -1))
        
    for i in range(rows):
        for j in range(cols):
            count, owner = board[i][j]
            if owner == ai_player:
                score -= count  # base score for owning orbs

                if (i, j) in edge_cells:
                    score -= 1

            elif owner and owner != ai_player:
                score += count 

    return score  

# Corner and Edge Controlling
def heuristic_7(board, ai_player):
    rows = len(board)
    cols = len(board[0])
    score = 0
    edge_cells = []
    corner_positions = [(0, 0), (0, cols - 1), (rows - 1, 0), (rows - 1, cols - 1)]

    
    # Edge consists of top, bottom, left, right cloumns excluding corners
    # Top & Bottom
    for j in range(1, cols - 1):
        edge_cells.append((0,j))
        edge_cells.append((rows - 1, j))
        
    # Left & Right
    for i in range(1, rows-1):
        edge_cells.append((i,0))
        edge_cells.append((i, cols -1))
        
    for i in range(rows):
        for j in range(cols):
            count, owner = board[i][j]
            if owner == ai_player:
                score += count  # base score for owning orbs

                if (i, j) in edge_cells:
                    score += 1
                elif (i,j) in corner_positions:
                    score += 2

            elif owner and owner != ai_player:
                score -= count 

    return score


def heuristic_8(board, ai_player):
    rows = len(board)
    cols = len(board[0])
    score = 0
    edge_cells = []
    corner_positions = [(0, 0), (0, cols - 1), (rows - 1, 0), (rows - 1, cols - 1)]

    
    # Edge consists of top, bottom, left, right cloumns excluding corners
    # Top & Bottom
    for j in range(1, cols - 1):
        edge_cells.append((0,j))
        edge_cells.append((rows - 1, j))
        
    # Left & Right
    for i in range(1, rows-1):
        edge_cells.append((i,0))
        edge_cells.append((i, cols -1))
        
    for i in range(rows):
        for j in range(cols):
            count, owner = board[i][j]
            if owner == ai_player:
                score += count  # base score for owning orbs
                
                if (i, j) in edge_cells:
                    score += 1
                elif (i,j) in corner_positions:
                    score += 2

            elif owner and owner != ai_player:
                score -= count 
                
                if (i, j) in edge_cells:
                    score -= 1
                elif (i,j) in corner_positions:
                    score -= 2


    return score


def heuristic_9(board, ai_player):
    rows = len(board)
    cols = len(board[0])
    score = 0
    edge_cells = []
    corner_positions = [(0, 0), (0, cols - 1), (rows - 1, 0), (rows - 1, cols - 1)]

    
    # Edge consists of top, bottom, left, right cloumns excluding corners
    # Top & Bottom
    for j in range(1, cols - 1):
        edge_cells.append((0,j))
        edge_cells.append((rows - 1, j))
        
    # Left & Right
    for i in range(1, rows-1):
        edge_cells.append((i,0))
        edge_cells.append((i, cols -1))
        
    for i in range(rows):
        for j in range(cols):
            count, owner = board[i][j]
            if owner == ai_player:
                score -= count  # base score for owning orbs
                
                if (i, j) in edge_cells:
                    score -= 1
                elif (i,j) in corner_positions:
                    score -= 2

            elif owner and owner != ai_player:
                score += count 
                
                if (i, j) in edge_cells:
                    score += 1
                elif (i,j) in corner_positions:
                    score += 2


    return score

def heuristic_10(board, ai_player):
    rows = len(board)
    cols = len(board[0])
    score = 0
    
    # Position weights
    corner_weight = 1.3
    edge_weight = 1.2
    center_weight = 0.8
    
    for i in range(rows):
        for j in range(cols):
            count, owner = board[i][j]
            position_modifier = 1.0  # Default for center cells
            
            # Check if corner or edge
            is_corner = (i == 0 or i == rows-1) and (j == 0 or j == cols-1)
            is_edge = (i == 0 or i == rows-1 or j == 0 or j == cols-1) and not is_corner
            
            if is_corner:
                position_modifier = corner_weight
            elif is_edge:
                position_modifier = edge_weight
            else:
                position_modifier = center_weight
            
            if owner == ai_player:
                score += count * position_modifier
            elif owner and owner != ai_player:
                score -= count * (1.0 + (position_modifier - 1.0)/2)  # Less penalty for opponent
    
    # Adding small randomness to prevent perfect predictability
    score += random.uniform(-0.5, 0.5)
    
    return score


heuristics = {
    1: heuristic_1,
    2: heuristic_2,
    3: heuristic_3,
    4: heuristic_4,
    5: heuristic_5,
    6: heuristic_6,
    7: heuristic_7,
    8: heuristic_8,
    9: heuristic_9,
    10: heuristic_10,
}

def get_heuristic_value(heuristic_id, board, player):
    heuristic_func = heuristics.get(heuristic_id)
    if heuristic_func is None:
        raise ValueError(f"Unknown heuristic id: {heuristic_id}")
    return heuristic_func(board, player)
 