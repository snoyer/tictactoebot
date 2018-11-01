

LINES = [
    (0,1,2), (3,4,5), (6,7,8),
    (0,3,6), (1,4,7), (2,5,8),
    (0,4,8), (2,4,6),
]

def winning_player(board):
    for ks in LINES:
        for player in ['x', 'o']:
            if all(board[k]==player for k in ks):
                return player
    return None

def winning_line(board):
    for ks in LINES:
        for player in ['x', 'o']:
            if all(board[k]==player for k in ks):
                return ks, player
    return None


def next_states(state):
    board, player = state
    for pos,c in enumerate(board):
        move = player, pos
        if c == '-' or c == ' ':
            new_board = board[:pos] + player + board[pos+1:]
            new_player = 'o' if player=='x' else 'x'
            yield move, (new_board, new_player)


def next_board(board, move):
    player, pos = move
    return board[:pos] + player + board[pos+1:]


def minimax(state, max_player, path=[]):
    board, _ = state
    depth = len(path)
    
    winner = winning_player(board)
    if winner:
        if winner==max_player:
            return path, (+1, -depth)
        else:
            return path, (-1, +depth)

    nexts = list(next_states(state))
    if nexts:
        f = max if depth%2==0 else min
        return f([minimax(s, max_player, path+[m]) for m,s in nexts], key=lambda x:x[1])
    
    return path, (0,0)



def best_move(state):
    moves, score = minimax(state, state[1])
    return moves[0] if moves else None


def guess_state(board):
    if '?' in board:
        return None
    
    x_count = board.count('x')
    o_count = board.count('o')

    if x_count > o_count:
        return board, 'o'
    elif o_count > x_count:
        return board, 'x'


def play_board(board):
    state = guess_state(board)
    if state:
        return best_move(state)

