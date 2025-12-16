import torch
from env import CheckersEnv
from dqn_agent import DQNAgent

env = CheckersEnv()
agent = DQNAgent(
    state_size=env.board_size * env.board_size,
    action_size=env.action_size
)

# ❗ เปลี่ยน model → q_network (หรือชื่อที่คุณใช้จริง)
agent.policy_net.load_state_dict(
    torch.load("checkers_ai.pth", map_location="cpu")
)
agent.policy_net.eval()

agent.epsilon = 0.0  # ไม่สุ่ม

state = env.get_state_tensor()
done = False

while not done:
    valid_actions = env.valid_actions()
    if not valid_actions:
        break

    action = agent.act(state, valid_actions)
    _, _, done, _ = env.step(action)
    state = env.get_state_tensor()

    for row in env.board:
        print(row)
    print("-" * 20)
