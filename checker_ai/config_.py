# config.py

# ===== Board =====
BOARD_SIZE = 6

EMPTY = 0
OPP = -1
AI = 1

# ===== Training =====
MAX_EPISODES = 2000
MAX_STEPS = 200

# ===== DQN =====
GAMMA = 0.99
LR = 1e-4
EPSILON_START = 1.0
EPSILON_END = 0.0005
EPSILON_DECAY = 0.999995
