'''Agent impl for car racing'''

import numpy as np
from tensordict import TensorDict

import torch
import torch.nn as nn
import torch.optim as optim

from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage

class ExpDecaySchedule:
    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay
        self.t = 1
        self.vs = []

    def __call__(self):
        value = self.end + (self.start - self.end) * np.exp(-self.decay * self.t)
        self.vs.append(value)
        return value

    def step(self):
        self.t += 1

class Policy:
    def __call__(self, state, Q):
        raise NotImplementedError

class EGreedyPolicy(Policy):

    def __init__(self, epsilon_schedule, n_actions):
        self.epsilon_schedule = epsilon_schedule
        self.n_actions = n_actions
        self.epsilon = epsilon_schedule()

    def __call__(self, state, Q):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.n_actions)
        else:
            state = torch.tensor(
                state,
                dtype=torch.float32,
                device='cuda'
            ).unsqueeze(0)
            return Q(state).argmax().item()

    def step(self, *args):
        self.epsilon_schedule.step()
        self.epsilon = self.epsilon_schedule()

class GreedyPolicy(Policy):
        def __call__(self, states, Q):
            return Q(states).argmax().item()

class DQN(nn.Module):
    def __init__(self, state_shape, n_actions):
        super(DQN, self).__init__()
        channels, height, width = state_shape
        self.in_channels = channels
        self.in_height = height
        self.in_width = width
        self.n_actions = n_actions
        
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2592, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
        )

    def forward(self, states):
        return self.net(states)

class Buffer:
    def __init__(self):
        self.buffer = TensorDictReplayBuffer(storage=LazyMemmapStorage(1e5))
        # when sample is called, the data is moved to the GPU
        self.buffer.append_transform(lambda x: x.to("cuda"))
    
    def __len__(self):
        return len(self.buffer)

    def push(self, s, a, r, s_, done):
        self.buffer.add(
            TensorDict({
                "s": torch.tensor(s),
                "a": torch.tensor(a),
                "s_": torch.tensor(s_),
                "r": torch.tensor(r),
                "done": torch.tensor(done)
        }))

    def sample(self, batch_size):
        batch = self.buffer.sample(batch_size)
        s = batch.get('s').squeeze().float()
        s_ = batch.get('s_').squeeze().float()
        a = batch.get('a').squeeze()
        r = batch.get('r').squeeze()
        done = batch.get('done').squeeze()
        return s, a, r, s_, done

class Agent:

    def __init__(
        self,
        state_shape,
        n_actions
        ):

        self.gamma = 0.99
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.batch_size = 32
        self.Q = DQN(state_shape, n_actions).to("cuda")
        self.Q_target = DQN(state_shape, n_actions).to("cuda")
        self.buffer = Buffer()
        self.e_greedy_policy = EGreedyPolicy(
            ExpDecaySchedule(1.0, 0.05, 0.0005),
            n_actions
        )
        self.greedy_policy = GreedyPolicy()
        self.optimizer = optim.AdamW(self.Q.parameters(), lr=1e-4, amsgrad=True)
        self.criterion = nn.SmoothL1Loss()

    def act(self, state):
        action = self.e_greedy_policy(state, self.Q)
        self.e_greedy_policy.step()
        return action

    def update(self):
        states, actions, rewards, new_states, terminateds = self.buffer.sample(self.batch_size)
        Q_values = self.Q(states)
        Q_values = Q_values[np.arange(self.batch_size), actions]
        with torch.no_grad():
            Q_target_values = self.Q_target(new_states).max(1).values
            Q_target_values[terminateds] = 0
            Q_target_values = rewards + self.gamma * Q_target_values
        loss = self.criterion(Q_values, Q_target_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        torch.nn.utils.clip_grad_value_(self.Q.parameters(), 100)
        loss = loss.item()
        return loss
    
    def sync_target(self):
        self.Q_target.load_state_dict(self.Q.state_dict())
        
    def save_checkpoint(self):
        file_path = "checkpoints/dqn_car_racing.pth"
        torch.save({
            'gamma': self.gamma,
             'Q': self.Q.state_dict(),
             'Q_target': self.Q_target.state_dict(),
             'optimizer': self.optimizer.state_dict(),
             'epsilon_schedule': {
                't': self.e_greedy_policy.epsilon_schedule.t,
                'start': self.e_greedy_policy.epsilon_schedule.start,
                'end': self.e_greedy_policy.epsilon_schedule.end,
                'decay': self.e_greedy_policy.epsilon_schedule.decay,
                'vs': self.e_greedy_policy.epsilon_schedule.vs
                },
             'state_shape': self.state_shape,
             'n_actions': self.n_actions,
            'batch_size': self.batch_size
         }, "checkpoints/dqn_car_racing.pth")
        self.buffer.buffer.dumps('checkpoints')
        print(f"Checkpoint saved to {file_path}")

    def load_checkpoint(self):
        checkpoint = torch.load('checkpoints/dqn_car_racing.pth', weights_only=False)
        self.gamma = checkpoint['gamma']
        self.Q.load_state_dict(checkpoint['Q'])
        self.Q_target.load_state_dict(checkpoint['Q_target'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.e_greedy_policy.epsilon_schedule.t = checkpoint['epsilon_schedule']['t']
        self.e_greedy_policy.epsilon_schedule.start = checkpoint['epsilon_schedule']['start']
        self.e_greedy_policy.epsilon_schedule.end = checkpoint['epsilon_schedule']['end']
        self.e_greedy_policy.epsilon_schedule.decay = checkpoint['epsilon_schedule']['decay']
        self.e_greedy_policy.epsilon_schedule.vs = checkpoint['epsilon_schedule']['vs']
        self.state_shape = checkpoint['state_shape']
        self.n_actions = checkpoint['n_actions']
        self.batch_size = checkpoint['batch_size']
        self.buffer.buffer.loads('checkpoints')
        print("Checkpoint loaded from checkpoints/dqn_car_racing.pth")
