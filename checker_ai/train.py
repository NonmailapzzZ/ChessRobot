import torch
import random
import numpy as np
from checkers_env import CheckersEnv
from dqn_model import DQN

env = CheckersEnv()
model = DQN()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss()

gamma = 0.99
epsilon = 1.0

for episode in range(2000):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        if random.random() < epsilon:
            move = random.choice(env.valid_moves())
        else:
            with torch.no_grad():
                q_values = model(state_t)[0]
            moves = env.valid_moves()
            move = max(
                moves,
                key=lambda m: q_values[m[1][0]*8 + m[1][1]]
            )

        next_state, reward, done = env.step(move)
        total_reward += reward

        # target Q
        next_t = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            max_next_q = model(next_t).max()

        target = reward + gamma * max_next_q

        pred = model(state_t)[0][move[1][0]*8 + move[1][1]]

        loss = loss_fn(pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        state = next_state

    epsilon = max(0.05, epsilon*0.995)

    if episode % 100 == 0:
        print(f"Episode {episode}, reward {total_reward}")
