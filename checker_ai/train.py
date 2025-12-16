from env import CheckersEnv
from dqn_agent import DQNAgent
from config import MAX_EPISODES, MAX_STEPS


def train():
    env = CheckersEnv()
    agent = DQNAgent(
        state_size=env.board_size * env.board_size,
        action_size=env.action_size
    )

    win_count = 0

    for ep in range(MAX_EPISODES):
        state = env.reset()
        done = False
        reward = 0

        while not done:
            valid_actions = env.valid_actions()
            if len(valid_actions) == 0:
                reward = -10.0
                done = True
                agent.remember(state, None, reward, None, done)
                break

            action = agent.act(state, valid_actions)
            next_state, reward, done, _ = env.step(action)

            agent.remember(state, action, reward, next_state, done)
            agent.learn()

            state = next_state

        if reward > 0:
            win_count += 1

        if (ep + 1) % 100 == 0:
            print(
                f"[Episode {ep+1}] Win rate: {win_count/100:.2f} | epsilon={agent.epsilon:.3f}"
            )
            win_count = 0

    agent.save("checkers_ai.pth")


if __name__ == "__main__":
    train()
