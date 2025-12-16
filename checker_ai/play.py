import torch
from checkers_env import CheckersEnv
from dqn_model import DQN

env = CheckersEnv()
model = DQN()
model.load_state_dict(torch.load("model.pth"))
model.eval()

state = env.reset()

move = max(
    env.valid_moves(),
    key=lambda m: model(
        torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    )[0][m[1][0]*8 + m[1][1]]
)

from_cell = move[0][0]*8 + move[0][1]
to_cell = move[1][0]*8 + move[1][1]

# เรียกแขนกล
move_robot(from_cell, to_cell)
