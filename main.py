import pygame
import sys
import time
import pandas as pd
from collections import deque
from heuristics import *
from random_agent import get_random_move
import copy
import random
pygame.init()


STATS_FILE = "venv/game_stats.csv"
heuristic_stats = {} # Format: {heuristic_num: {"wins": int, "games": int, "durations": [float]}}
game_start_time = 0

ai_heuristics = {}
# Constants
ROWS, COLS = 9, 6
CELL_SIZE = 80
WIDTH, HEIGHT = COLS * CELL_SIZE, ROWS * CELL_SIZE
FPS = 60  # frames per sec
MAX_DEPTH = 3
# works good with 250 explosions
MAX_EXPLOSIONS = 300 
MAX_ORBS = 300

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (220, 20, 60)
BLUE = (30, 144, 255)
GREY = (200, 200, 200)

# Colors for menu
DARK_GREY = (50, 50, 50)
LIGHT_GREY = (180, 180, 180)
HOVER_COLOR = (57, 255, 16)
BUTTON_COLOR = (50, 50, 50)


# Font sizes
TITLE_FONT = pygame.font.SysFont("Monospace", 48)
BUTTON_FONT = pygame.font.SysFont("Consolas", 28)
MENU_FONT = pygame.font.SysFont("Monospace", 21)  # Added MENU_FONT definition

# AI is ALWAYS R 
# game should check for victory only after both have started playing
player_alive = {'R': False, 'B': False}
player_type = {'R': 'Human', 'B': 'AI'}  # default
game_mode_chosen = False

HEURISTIC_BUTTON_WIDTH = 60
HEURISTIC_BUTTON_HEIGHT = 40
HEURISTIC_BUTTON_MARGIN = 10
HEURISTIC_FONT = pygame.font.SysFont("Consolas", 20)



def init_stats():
    # Initializing statistics from file"""
    global heuristic_stats
    try:
        df = pd.read_csv(STATS_FILE)
        for _, row in df.iterrows():
            heuristic_stats[row['heuristic']] = {
                "wins": row['wins'],
                "games": row['games'],
                "durations": eval(row['durations'])  # Convert string to list
            }
    except:
        # Initializing empty stats for all heuristics (1-10)
        for i in range(1, 11):
            heuristic_stats[i] = {"wins": 0, "games": 0, "durations": []}




def save_stats():
    data = []
    for heuristic, stats in heuristic_stats.items():
        data.append({
            "heuristic": heuristic,
            "wins": stats["wins"],
            "games": stats["games"],
            "durations": str(stats["durations"]),
            "win_rate": stats["wins"]/stats["games"] if stats["games"] > 0 else 0,
            "avg_duration": sum(stats["durations"])/len(stats["durations"]) if stats["durations"] else 0
        })
    pd.DataFrame(data).to_csv(STATS_FILE, index=False)


# Initialize board (each cell: [orb_count, 'R'/'B'/None])
board = [[[0, None] for _ in range(COLS)] for _ in range(ROWS)]
current_player = 'B'  # R = Red, B = Blue; B plays first


screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 36)

def draw_board():
    screen.fill(BLACK)  # black background

    # Draw grid lines (green 3D-like)
    for r in range(ROWS):
        for c in range(COLS):
            x = c * CELL_SIZE
            y = r * CELL_SIZE
            pygame.draw.rect(screen, (0, 255, 0), (x, y, CELL_SIZE, CELL_SIZE), 2)

            orb_count, owner = board[r][c]
            if orb_count > 0 and owner is not None:
                # color = RED if owner == 'R' else BLUE
                if owner == 'R':
                    color = RED
                elif owner == 'B':
                    color = BLUE 
                  

                center_x = x + CELL_SIZE // 2
                center_y = y + CELL_SIZE // 2
                offset = 10

                if orb_count == 1:
                    pygame.draw.circle(screen, color, (center_x, center_y), 12)
                elif orb_count == 2:
                    pygame.draw.circle(screen, color, (center_x - offset, center_y), 12)
                    pygame.draw.circle(screen, color, (center_x + offset, center_y), 12)
                else:  # 3 or more
                    pygame.draw.circle(screen, color, (center_x, center_y - offset), 12)
                    pygame.draw.circle(screen, color, (center_x - offset, center_y + offset), 12)
                    pygame.draw.circle(screen, color, (center_x + offset, center_y + offset), 12)


def get_opponent(player):
    return 'R' if player == 'B' else 'B'


        
def get_critical_mass(x, y):
    neighbors = 0
    for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < ROWS and 0 <= ny < COLS:
            neighbors += 1
    return neighbors

########################################################################### EXPLODE ###############################################################################################
def explode(board, x, y, caller):
    explosion_count = 0
    queue = deque()
    queue.append((x, y))

    while queue and explosion_count <= MAX_EXPLOSIONS:
        cx, cy = queue.popleft()  # Get the oldest node
        orb_count, owner = board[cx][cy]
        critical_mass_val = get_critical_mass(cx, cy)

        if orb_count < critical_mass_val:
            continue  # no explosion needed

        board[cx][cy] = (orb_count - critical_mass_val, None)
        if caller != "Simulator":
            print("Explosion!!!!!!!!")
             
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < ROWS and 0 <= ny < COLS:
                n_count, n_owner = board[nx][ny]

                if n_count == 0:
                  board[nx][ny] = (1, owner)
                  if board[nx][ny][0] >= get_critical_mass(nx, ny):
                    queue.append((nx, ny))
                    if caller != "Simulator":
                     print(f"Got color {owner} and 1 orb at coordinate {nx, ny}")
                else:
                  board[nx][ny] = (n_count + 1, owner)
                  if board[nx][ny][0] >= get_critical_mass(nx, ny):
                    queue.append((nx, ny))
                    if caller != "Simulator":
                     print(f"Got color {owner} and {n_count + 1} orbs at coordinate {nx, ny}")

                
                          
        explosion_count += 1
        if board is globals()['board']:  # only check for the main game board
            alive_players = set()
            for row in board:
                for count, cell_owner in row:
                    if cell_owner:
                        alive_players.add(cell_owner)

            print("Alive players mid-chain:", alive_players)
            if len(alive_players) == 1 and all(player_alive.values()):  # both have played
                winner = alive_players.pop()
                print(f"Winner found during chain explosion: {winner}")
                record_game_result(winner, ai_heuristics)
                if board is globals()['board']:  # only draw if it's the real board
                    draw_board()
                    pygame.display.update()
                    pygame.time.delay(90)
                show_game_over(winner)
                return
    
       
        
        # skip GUI update if it's a simulation
        if board is globals()['board']:  # only draw if it's the real board
            draw_board()
            pygame.display.update()
            pygame.time.delay(90)
    
    if explosion_count > MAX_EXPLOSIONS:
        print("Warning: Explosion chain too long!!!")
        
    



# For getting all possible moves for AI
def get_possible_moves(board, player):
    moves = []
    for r in range(ROWS):
        for c in range(COLS):
            count, owner = board[r][c]
            if owner is None or owner == player:
                moves.append((r, c))
    # print("length of moves : ", len(moves))
    # print('\n')            
    return moves


# AI simulates different moves to evaluate 
def simulate_move(board, move, player):
    new_board = copy.deepcopy(board)
    r, c = move
    count, owner = new_board[r][c]
    new_board[r][c] = (count + 1, player)

    if new_board[r][c][0] >= get_critical_mass(r, c):
        explode(new_board, r, c, "Simulator")
    return new_board



########################################################################### MINIMAX ############################################################################################### 
def minimax(board, depth, alpha, beta, maximizing, ai_player, heuristic_id): 
    # alpha = -inf; beta = +inf
    if depth == MAX_DEPTH:
        value = get_heuristic_value(heuristic_id, board, ai_player)
        return value, None
        # return heuristic_2(board, ai_player), None # stopping recursion when max depth reached

    moves = get_possible_moves(board, ai_player if maximizing else get_opponent(ai_player))
    # if len(moves) > 20:
    #     moves = random.sample(moves, 20)  # Limit branching factor
    if not moves:
        return heuristic_1(board, ai_player), None # either game over or dead state; choose winner heuristic-wise

    best_move = None

    if maximizing:
        max_eval = float('-inf')
        for move in moves:
            new_board = simulate_move(board, move, ai_player)
            eval, _ = minimax(new_board, depth + 1, alpha, beta, False, ai_player, heuristic_id)
            if eval > max_eval:
                max_eval = eval
                best_move = move
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval, best_move
    else:
        min_eval = float('inf')
        for move in moves:
            new_board = simulate_move(board, move, get_opponent(ai_player))
            eval, _ = minimax(new_board, depth + 1, alpha, beta, True, ai_player, heuristic_id)
            if eval < min_eval:
                min_eval = eval
                best_move = move
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval, best_move
    
    
def save_board_to_file(board, player_type):  # player_type: "Human" or "Ai"
    with open("gamestate.txt", "w") as f:
        f.write(f"{player_type} Move:\n")
        for row in board:
            line = []
            for count, owner in row:
                if count == 0 or owner is None:
                    line.append("0")
                else:
                    line.append(f"{count}{owner}")
            f.write(" ".join(line) + "\n")
   



########################################################################### AI MOVE ###############################################################################################
def ai_move(ai_player, heuristic_id):
    pygame.event.pump()
    legal_moves = get_possible_moves(board, ai_player)

    if not legal_moves:
        # Dead state, evaluate winner based on heuristic_1
        red_score = heuristic_1(board, 'R')
        blue_score = heuristic_1(board, 'B')
        winner = 'R' if red_score > blue_score else 'B'
        print("No legal moves available. Winner decided by heuristic_1.")
        show_game_over(winner)
        return

    start = time.time()
    score = None
    move = None

    try:
        score, move = minimax(board, 1, float('-inf'), float('inf'), True, ai_player, heuristic_id)
    except Exception as e:
        print(f"Error during AI computation: {e}")

    end = time.time()
    time_taken = end - start
    print(f"AI move took {time_taken:.2f} seconds")

    if time_taken > 4 or move is None:
        print("AI could not make decision within time limit")
        move = random.choice(legal_moves)

    # Make the move
    r, c = move
    count, _ = board[r][c]
    board[r][c] = (count + 1, ai_player)

    if board[r][c][0] >= get_critical_mass(r, c):
        explode(board, r, c, "AI")

    pygame.time.delay(900)
    save_board_to_file(board, "Ai")

    winner = check_game_over()
    if winner:
        print(f"{winner} won!")
        return
       # show_game_over(winner)

    global current_player
    current_player = get_opponent(ai_player)



########################################################################### RANDOM AGENT MOVE ###############################################################################################
def random_agent_move(agent_player):
    pygame.event.pump()
    move = get_random_move(board, agent_player)

    if not move:
        red_score = heuristic_1(board, 'R')
        blue_score = heuristic_1(board, 'B')
        winner = 'R' if red_score > blue_score else 'B'
        print("No legal moves. Winner by heuristic_1.")
        show_game_over(winner)
        return

    r, c = move
    count, _ = board[r][c]
    board[r][c] = (count + 1, agent_player)
    if board[r][c][0] >= get_critical_mass(r, c):
        explode(board, r, c, "Random")

    pygame.time.delay(900)
    save_board_to_file(board, "Random")
    winner = check_game_over()
    if winner:
        print(f"{winner} won!")
        return
     #   show_game_over(winner)

    global current_player
    current_player = get_opponent(agent_player)

    
    
        
def check_game_over():
    players = set()
    for row in board:
        for count, owner in row:
            if owner:
                players.add(owner)

    if 'R' in players:
        player_alive['R'] = True
    if 'B' in players:
        player_alive['B'] = True

    # Only check for win condition if both players have entered the game
    # The built-in function all() checks if every item in a list is True; here we need ([True, True])
    if all(player_alive.values()):
        if len(players) == 1:
            return players.pop()  # That one is the winner

    return None


def show_game_over(winner):
    font = pygame.font.SysFont(None, 72)
    text = font.render(f"{winner} won!", True, (255, 255, 0))
    rect = text.get_rect(center=(WIDTH // 2, HEIGHT // 2))
    screen.blit(text, rect)
    
    
    pygame.display.update()
    pygame.time.delay(3000)  # show for 3 seconds
    pygame.quit()
    sys.exit()



# def print_stats():
#     """Display stats in console"""
#     print("\nCurrent Statistics:")
#     print("Heuristic | Games | Wins | Win Rate | Avg Duration")
#     for h in sorted(heuristic_stats.keys()):
#         s = heuristic_stats[h]
#         wr = s["wins"]/s["games"] if s["games"] > 0 else 0
#         avg = sum(s["durations"])/len(s["durations"]) if s["durations"] else 0
#         print(f"{h:8} | {s['games']:5} | {s['wins']:4} | {wr:.2f} | {avg:.2f}s")


def record_game_result(winner, ai_heuristics):
    game_duration = time.time() - game_start_time
    
    # Update stats for both AI players (if applicable)
    for player in ['R', 'B']:
        if player_type[player] == 'AI':
            heuristic = ai_heuristics[player]
            heuristic_stats[heuristic]["games"] += 1
            heuristic_stats[heuristic]["durations"].append(game_duration)
            
            if winner == player:
                heuristic_stats[heuristic]["wins"] += 1
    
    save_stats()
    #print_stats()
  







def draw_button(text, rect, hovered):
    color = HOVER_COLOR if hovered else BUTTON_COLOR
    pygame.draw.rect(screen, color, rect, border_radius=12)
    pygame.draw.rect(screen, WHITE, rect, 3, border_radius=12)

    label = BUTTON_FONT.render(text, True, RED)
    label_rect = label.get_rect(center=rect.center)
    screen.blit(label, label_rect)



def draw_menu():
    screen.fill(BLACK)
    title = TITLE_FONT.render("Chain Reaction", True, (230,0,184))
    screen.blit(title, (WIDTH // 2 - title.get_width() // 2, 80))

    mouse_pos = pygame.mouse.get_pos()

    # Define button areas
    button_width = 300
    button_height = 50
    spacing = 30
    start_y = 200

    buttons = [
        ("Human vs Human", pygame.Rect(WIDTH // 2 - button_width // 2, start_y, button_width, button_height)),
        ("AI vs Human",    pygame.Rect(WIDTH // 2 - button_width // 2, start_y + button_height + spacing, button_width, button_height)),
        ("AI vs AI",       pygame.Rect(WIDTH // 2 - button_width // 2, start_y + 2 * (button_height + spacing), button_width, button_height)),
        ("AI vs Random",   pygame.Rect(WIDTH // 2 - button_width // 2, start_y + 3 * (button_height + spacing), button_width, button_height)),

    ]

    for text, rect in buttons:
        hovered = rect.collidepoint(mouse_pos)
        draw_button(text, rect, hovered)

    pygame.display.flip()
    return buttons


def draw_heuristic_selection(player_num, selected_heuristic=None, hovered_button=None):
    """Draw the heuristic selection screen with visual feedback"""
    screen.fill(BLACK)
    
    # Title
    title_str = f"Choose heuristic for Player {player_num} (AI)"
    title_text = MENU_FONT.render(title_str, True, WHITE)
    screen.blit(title_text, (WIDTH // 2 - title_text.get_width() // 2, 50))
    
    buttons = []
    
    # Layout constants
    start_x = WIDTH // 2 - (4 * (HEURISTIC_BUTTON_WIDTH + HEURISTIC_BUTTON_MARGIN)) // 2
    start_y = 150
    
    for i in range(10):
        heuristic_num = i + 1
        row = i // 4
        col = i % 4

        if row == 2 and col >= 3:  # Only 3 buttons in last row
            continue

        x_pos = start_x + col * (HEURISTIC_BUTTON_WIDTH + HEURISTIC_BUTTON_MARGIN)
        y_pos = start_y + row * (HEURISTIC_BUTTON_HEIGHT + HEURISTIC_BUTTON_MARGIN)

        rect = pygame.Rect(x_pos, y_pos, HEURISTIC_BUTTON_WIDTH, HEURISTIC_BUTTON_HEIGHT)
        
        # Detect state
        is_hovered = hovered_button == rect
        is_selected = selected_heuristic == heuristic_num
        
        # Draw button
        if is_selected:
            pygame.draw.rect(screen, HOVER_COLOR, rect, border_radius=5)
            pygame.draw.rect(screen, WHITE, rect, 3, border_radius=5)
        elif is_hovered:
            pygame.draw.rect(screen, LIGHT_GREY, rect, border_radius=5)
            pygame.draw.rect(screen, WHITE, rect, 2, border_radius=5)
        else:
            pygame.draw.rect(screen, GREY, rect, border_radius=5)
            pygame.draw.rect(screen, WHITE, rect, 1, border_radius=5)

        # Draw label
        label = HEURISTIC_FONT.render(str(heuristic_num), True, BLACK)
        screen.blit(label, (
            rect.centerx - label.get_width() // 2,
            rect.centery - label.get_height() // 2
        ))

        buttons.append((heuristic_num, rect))

    # Footer text
    instruction_text = BUTTON_FONT.render("Press ENTER to continue", True, WHITE)
    screen.blit(instruction_text, (
        WIDTH // 2 - instruction_text.get_width() // 2,
        HEIGHT - 80
    ))

    pygame.display.flip()
    return buttons


def handle_heuristic_selection(player_num):
    """Handle selection with visual feedback"""
    selected_heuristic = 10  # Default
    hovered_button = None
    
    while True:
        mouse_pos = pygame.mouse.get_pos()
        new_hover = None
        
        # Draw buttons with current states
        buttons = draw_heuristic_selection(player_num, selected_heuristic, hovered_button)
        
        # Check for hover
        for heuristic_num, rect in buttons:
            if rect.collidepoint(mouse_pos):
                new_hover = rect
                break
        
        # Update hover state if changed
        if new_hover != hovered_button:
            hovered_button = new_hover
            buttons = draw_heuristic_selection(player_num, selected_heuristic, hovered_button)
        
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                # Check for clicks
                for heuristic_num, rect in buttons:
                    if rect.collidepoint(mouse_pos):
                        selected_heuristic = heuristic_num
                        # Redraw to show selection
                        buttons = draw_heuristic_selection(player_num, selected_heuristic, hovered_button)
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                return selected_heuristic
        
        pygame.time.delay(30)  # Prevent CPU overuse

def draw_player_info_screen():
    screen.fill(BLACK)
    title = TITLE_FONT.render("Player Info", True, (230,0,184))
    screen.blit(title, (WIDTH // 2 - title.get_width() // 2, 80))


    role_map = {
        'Human': "HUMAN",
        'AI': "AI",
        'Random': "RANDOM"
    }

    colors = {'R': (255, 0, 0), 'B': (0, 100, 255)}  # red and blue
    role_map = {'Human': 'HUMAN', 'AI': 'AI', 'Random': 'RANDOM'}

    start_y = 180
    line_spacing = 80
    text_x = WIDTH // 2 - 150
    circle_radius = 15
    circle_offset_x = 250  # distance from text start

    for i, color in enumerate(['R', 'B']):
        role = player_type[color]
        role_label = role_map.get(role, '?')
        orb_color = colors[color]

        # Render and position text
        text = MENU_FONT.render(f"Player {i+1} ({role_label}):", True, WHITE)
        y = start_y + i * line_spacing
        screen.blit(text, (text_x, y))

        circle_x = text_x + circle_offset_x
        circle_y = y + text.get_height() // 2  # center vertically with text
        pygame.draw.circle(screen, orb_color, (circle_x, circle_y), circle_radius)
    instruction = MENU_FONT.render("Press ENTER to start the game", True, GREY)
    screen.blit(instruction, (WIDTH // 2 - instruction.get_width() // 2, HEIGHT - 100))

    pygame.display.flip()

    # Wait for ENTER key
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                waiting = False


    
 
def handle_menu_click(pos, buttons):
    global player_type, game_mode_chosen
    if buttons[0][1].collidepoint(pos):
        player_type = {'R': 'Human', 'B': 'Human'}
        pygame.display.set_caption("Chain Reaction - Human vs Human")
        game_mode_chosen = True
    elif buttons[1][1].collidepoint(pos):
        player_type = {'R': 'AI', 'B': 'Human'}
        pygame.display.set_caption("Chain Reaction - AI vs Human")
        game_mode_chosen = True
    elif buttons[2][1].collidepoint(pos):
        player_type = {'R': 'AI', 'B': 'AI'}
        pygame.display.set_caption("Chain Reaction - AI vs AI")
        game_mode_chosen = True
    elif buttons[3][1].collidepoint(pos):
        player_type = {'R': 'AI', 'B': 'Random'}
        pygame.display.set_caption("Chain Reaction - AI vs Random Move Agent")
        game_mode_chosen = True  
    # if game_mode_chosen:
    #     draw_player_info_screen()      
        

        


########################################################################### HUMAN MOVE ###############################################################################################
def handle_click(pos):
    global current_player

    col = pos[0] // CELL_SIZE
    row = pos[1] // CELL_SIZE

    if row < 0 or row >= ROWS or col < 0 or col >= COLS:
        return

    orb_count, owner = board[row][col]
    if owner is None or owner == current_player:
        board[row][col] = (orb_count + 1, current_player)
        if board[row][col][0] >= get_critical_mass(row, col):
            explode(board, row, col, "Human")
        
        winner = check_game_over()
        if winner:
            print(f"{winner} won!")
            return
           # show_game_over(winner)

        current_player = get_opponent(current_player)
        
  

def main():
    global current_player, game_mode_chosen, player_type, game_start_time
    init_stats()  # Initialize statistics tracking
    
    # Main menu loop
    while not game_mode_chosen:
        buttons = draw_menu()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                handle_menu_click(pygame.mouse.get_pos(), buttons)
    
    # Heuristic selection for AI players
    global ai_heuristics
    ai_heuristics = {'R': 10, 'B': 10}  # Default to heuristic 10
    
    if player_type['R'] == 'AI':
        ai_heuristics['R'] = handle_heuristic_selection(1)
        print(f"R AI has chosen {ai_heuristics['R']}")
    
    if player_type['B'] == 'AI':
        ai_heuristics['B'] = handle_heuristic_selection(2)
        print(f"B AI has chosen {ai_heuristics['B']}")
    
    # Now draw player info screen or proceed to game
    draw_player_info_screen()
    game_start_time = time.time()  # Start game timer
    
    # Main Game Loop
    while True:
        clock.tick(FPS)

        if player_type[current_player] == 'AI':
            ai_move(current_player, ai_heuristics[current_player])
            
        elif player_type[current_player] == 'Random':
            random_agent_move(current_player)
        else:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    save_stats()  # Save before quitting
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    handle_click(pygame.mouse.get_pos())

        # Check for game over condition
        winner = check_game_over()
        if winner:
            # Record results before showing game over
            record_game_result(winner, ai_heuristics)
            show_game_over(winner)

        draw_board()
        pygame.display.flip()

if __name__ == "__main__":
    main()
