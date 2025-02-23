import gymnasium as gym
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tqdm
import pandas as pd

env = gym.make("Blackjack-v1", render_mode=None, sab=True)

class Policy:
    def __init__(self, mode="train"):
        self.mode = mode

    def __call__(self, s):
        raise NotImplementedError

    def update(self, *args):
        pass

class RandomPolicy(Policy):
    
    def __call__(self, s):
        return np.random.choice([0, 1])

class BlackjackTabularQ:
    def __init__(self):
        self.Q = np.zeros((17, 10, 2, 2))

    def __getitem__(self, s):
        return self.Q[(s[0] - 5, s[1] -1) + s[2:]]

class DefaultSchedule:
    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay
        self.t = 0
        self.vs = []

    def __call__(self):
        lr = self.end + (self.start - self.end) * np.exp(-self.decay * self.t)
        self.vs.append(lr)
        self.t += 1
        return lr

class EpsilonSchedule(DefaultSchedule):
    def __init__(self, start=1.0, end=0.00005, decay=0.00005):
        super().__init__(start, end, decay)

class LRSchedule(DefaultSchedule):
    def __init__(self, start=1.0, end=0.00005, decay=0.00005):
        super().__init__(start, end, decay)

class FixedLRSchedule:
    def __init__(self, start):
        self.start = start

    def __call__(self):
        return self.start

class FixedEpsilonSchedule:
    def __init__(self, start):
        self.start = start

    def __call__(self):
        return self.start

class EpsilonGreedy:
    
    def __init__(self, e_schedule):
        self.e_schedule = e_schedule
        self.e = e_schedule()

    def __call__(self, Q, s):
        if np.random.random() < self.e:
            return env.action_space.sample()
        else:
            return np.argmax(Q[s])
    
    def update(self, *args):
        self.e = self.e_schedule()

class Boltzmann:
        
    def __init__(self, beta):
        self.beta = beta

    def __call__(self, Q, s):
        p = np.exp(self.beta * Q[s])
        # if not np.isclose(np.sum(Q[s]), 0):
        #     print(Q[s])
        p = p / np.sum(p)
        return np.random.choice(np.arange(len(p)), p=p)

    def update(self, *args):
        pass

class UCB1:

    def __init__(self):
        self.c = 1
        self.t = 1
        self.N = BlackjackTabularQ()

    def __call__(self, Q, s):
        # print(self.t)
        N_s = self.N[s]
        if np.all(N_s == 0):
            return np.random.choice([0, 1])
        if np.any(N_s == 0):
            return np.argmax(N_s == 0)
        ucb = self.c * np.sqrt(np.log(self.t) / N_s)
        return np.argmax(Q[s] + ucb)

    def update(self, s, a):
        self.N[s][a] += 1
        self.t += 1

class QLearningPolicy(Policy):
    
    def __init__(self, base_policy, lr_schedule=FixedLRSchedule(0.1), g=0.99, **kwargs):
        self.Q = BlackjackTabularQ()
        self.base_policy = base_policy
        self.lr_schedule = lr_schedule
        self.lr = lr_schedule()
        self.g = g
        self.training_error = []
        super().__init__(**kwargs)

    def __call__(self, s):
        return self.base_policy(self.Q, s)

    def update(self, s, a, r, s_, done):
        if done:
            self.base_policy.update(s, a)
            self.lr = self.lr_schedule()
            Q_s_ = 0
        else:
            Q_s_ = np.max(self.Q[s_])
        self.Q[s][a] += \
            self.lr * (r + self.g * Q_s_ - self.Q[s][a])

def run_round(p, mode):
    s, _ = env.reset()
    p.mode = mode

    done = False
    while not done:
        a = p(s)
        s_, r, done, _, _ = env.step(a)
        if mode == "train":
            p.update(s, a, r, s_, done)
        s = s_
    return p, r

def train(p, num_rounds):
    rs = []
    for _ in tqdm.tqdm(range(num_rounds)):
        p, r = run_round(p, mode="train")
        rs.append(r)
    return p, rs

def test(p, num_rounds):
    rewards = []
    for _ in tqdm.tqdm(range(num_rounds)):
        rewards.append(run_round(p, mode="test")[1])
    return rewards

def run():
    epochs = 100000

    policies = [
        {
            'name': 'Epsilon Greedy e=0.1',
            'p': QLearningPolicy(
                base_policy=EpsilonGreedy(FixedEpsilonSchedule(0.1)),
                lr_schedule=FixedLRSchedule(0.01)
            ),
            'rewards': []
        },
        {
            'name': 'Epsilon Greedy e=0.001',
            'p': QLearningPolicy(
                base_policy=EpsilonGreedy(FixedEpsilonSchedule(0.001)),
                lr_schedule=FixedLRSchedule(0.01)
            ),
            'rewards': []
        },
        # {
        #     'name': 'Boltzmann beta=100',
        #     'p': QLearningPolicy(
        #         base_policy=Boltzmann(100),
        #         lr_schedule=FixedLRSchedule(0.01)
        #     ),
        #     'rewards': []
        # },
        {
            'name': 'Boltzmann beta=10',
            'p': QLearningPolicy(
                base_policy=Boltzmann(10),
                lr_schedule=FixedLRSchedule(0.01)
            ),
            'rewards': []
        },
        {
            'name': 'UCB1',
            'p': QLearningPolicy(
                base_policy=UCB1(),
                lr_schedule=FixedLRSchedule(0.01)
            ),
            'rewards': []
        },
        {
            'name': 'Epsilon Greedy with Exp. Decay',
            'p': QLearningPolicy(
                base_policy=EpsilonGreedy(EpsilonSchedule()),
                lr_schedule=FixedLRSchedule(0.01)
            ),
            'rewards': []
        }
    ]

    # for p in policies:
    #     pol = p['p']
    #     bp = pol.base_policy
    #     # if bp has attr t, print t
    #     if hasattr(bp, "t"):
    #         print(f"UCB1 t: {bp.t}")
        

    for p in policies:
        p['p'], p['rewards'] = train(p['p'], epochs)

    for p in policies:
        p['rewards'] = np.convolve(p['rewards'], np.ones(10000), mode="valid") / 10000

    for p in policies:
        p['df'] = pd.DataFrame({
            "t": range(len(p['rewards'])),
            "Average Reward": p['rewards'],
            "Policy": p['name']
        })

    df = pd.concat([p['df'] for p in policies])

    fig, axes = plt.subplots()

    sns.lineplot(data=df, x="t", y="Average Reward", hue="Policy", ax=axes)

    plt.savefig("blackjack_extra_credit.png")



run()
env.close()
