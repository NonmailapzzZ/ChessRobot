import torch
from checker_ai.env import CheckersEnv
from checker_ai.dqn_agent import DQNAgent
from checker_ai.utils import action_to_move
from checker_ai.config import BOARD_SIZE, AI, OPP

def predict_move(board_list, model_path="checker_ai.pth"):
    """
    board_list: 2D list จากกล้อง
    return: {"from": (r,c), "to": (r,c)}
    """

    # สร้าง env เปล่า
    env = CheckersEnv()
    env.board = board_list
    env.current_player = AI

    state = torch.tensor(board_list, dtype=torch.float32).unsqueeze(0)

    agent = DQNAgent(
        state_size=BOARD_SIZE * BOARD_SIZE,
        action_size=env.action_size
    )

    agent.q_net.load_state_dict(torch.load(model_path))
    agent.epsilon = 0.0  # ไม่สุ่ม

    valid_actions = env.valid_actions()
    if len(valid_actions) == 0:
        return None  # ไม่มีตาเดิน

    action = agent.act(state, valid_actions)
    from_pos, to_pos = action_to_move(action)

    return {
        "from": from_pos,
        "to": to_pos
    }
