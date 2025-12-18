import torch
from checker_ai.config_ import BOARD_SIZE


def board_to_tensor(board):
    return torch.tensor(board, dtype=torch.float32).unsqueeze(0)


def move_to_action(move):
    (r1, c1), (r2, c2) = move
    return ((r1 * BOARD_SIZE + c1) * BOARD_SIZE + r2) * BOARD_SIZE + c2


def action_to_move(action):
    c2 = action % BOARD_SIZE
    action //= BOARD_SIZE
    r2 = action % BOARD_SIZE
    action //= BOARD_SIZE
    c1 = action % BOARD_SIZE
    r1 = action // BOARD_SIZE
    return (r1, c1), (r2, c2)