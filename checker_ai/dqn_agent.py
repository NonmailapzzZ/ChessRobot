import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from config import GAMMA, LR, EPSILON_DECAY, EPSILON_END

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(state_size, 256),
            nn.ReLU(),
            nn.Linear(256, action_size)
        )

    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.policy_net = QNetwork(state_size, action_size)
        self.target_net = QNetwork(state_size, action_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.loss_fn = nn.MSELoss()

        self.memory = deque(maxlen=10000)
        self.epsilon = 1.0
        self.action_size = action_size
        self.learn_step = 0
        self.target_update_freq = 500

    def act(self, state, valid_actions):
        if random.random() < self.epsilon:
            return random.choice(valid_actions)

        with torch.no_grad():
            q = self.policy_net(state)[0]

        return max(valid_actions, key=lambda a: q[a].item())

    def remember(self, s, a, r, s2, done):
        self.memory.append((s, a, r, s2, done))


    def learn(self, batch_size=64):
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)

        for s, a, r, s2, done in batch:
            with torch.no_grad():
                q_next = torch.max(self.target_net(s2)).item()
                q_target = r if done else r + GAMMA * q_next

            q_pred = self.policy_net(s)[0][a]
            loss = self.loss_fn(
                q_pred,
                torch.tensor(q_target, dtype=torch.float32)
            )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.learn_step += 1
        if self.learn_step % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)



    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path, device="cpu"):
        self.policy_net.load_state_dict(
            torch.load(path, map_location=device)
        )
        self.policy_net.eval()


