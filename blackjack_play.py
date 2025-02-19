import gymnasium as gym
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tqdm
import pandas as pd
from collections import defaultdict

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

class Schedule:
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

class EpsilonSchedule(Schedule):
    def __init__(self, start=1.0, end=0.00005, decay=0.00005):
        super().__init__(start, end, decay)

class LRSchedule(Schedule):
    def __init__(self, start=1.0, end=0.00005, decay=0.00005):
        super().__init__(start, end, decay)

class QLearningPolicy(Policy):
    
    def __init__(self, e_schedule=EpsilonSchedule(), lr_schedule=LRSchedule(), g=0.99, **kwargs):
        self.Q = BlackjackTabularQ()
        self.e_schedule = e_schedule
        self.lr_schedule = lr_schedule
        self.e = e_schedule()
        self.lr = lr_schedule()
        self.g = g
        self.training_error = []
        super().__init__(**kwargs)

    def __call__1(self, s):
        if self.mode == "train" and np.random.rand() < self.e:
            return np.random.choice([0, 1])
        else:
            return np.argmax(self.Q[s])

    def __call__(self, s):
        if np.random.random() < self.e:
            return env.action_space.sample()
        else:
            return np.argmax(self.Q[s])

    def update(self, s, a, r, s_, done):
        if done:
            self.e = self.e_schedule()
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

    p, rewards = train(QLearningPolicy(), epochs)
    rewards2 = test(RandomPolicy(), epochs)
    rewards = np.convolve(rewards, np.ones(10000), mode="valid") / 10000
    rewards2 = np.convolve(rewards2, np.ones(10000), mode="valid") / 10000

    df_sch = pd.DataFrame({
        "t": range(len(p.e_schedule.vs)),
        "Epsilon": p.e_schedule.vs,
        "Learning rate": p.lr_schedule.vs
    })

    fig, axes = plt.subplots(2)

    ax2 = axes[1]

    sns.lineplot(data=df_sch, x="t", y="Epsilon", ax=ax2)
    sns.lineplot(data=df_sch, x="t", y="Learning rate", ax=ax2)

    ax2.set_xlabel("t")
    ax2.set_ylabel("Epsilon and Learning rate")

    df = pd.DataFrame({
        "t": range(len(rewards)),
        "Average Reward": rewards,
        "Policy": "Q-learning"
    })
    df2 = pd.DataFrame({
        "t": range(len(rewards2)),
        "Average Reward": rewards2,
        "Policy": "Random"
    })
    df = pd.concat([df, df2])
    ax1 = axes[0]
    sns.lineplot(data=df, x="t", y="Average Reward", ax=ax1, hue="Policy")
    plt.savefig("blackjack.png")

    q = p.Q.Q
    q = q.copy()
    q_stand_ace = q[:, :, 1, 0]
    q_hit_ace = q[:, :, 1, 1]
    q_stand_no_ace = q[:, :, 0, 0]
    q_hit_no_ace = q[:, :, 0, 1]

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    new_labels_x = [str(i) for i in range(5, 22)]

    new_labels_y = [str(i) for i in range(1, 11)]

    vmin = np.min([q_stand_ace, q_hit_ace, q_stand_no_ace, q_hit_no_ace])
    vmax = np.max([q_stand_ace, q_hit_ace, q_stand_no_ace, q_hit_no_ace])

    sns.heatmap(q_stand_ace, ax=axes[0, 0], xticklabels=new_labels_y, yticklabels=new_labels_x, vmin=vmin, vmax=vmax, cbar=False, annot=True, fmt=".2f")
    axes[0, 0].set_title("Q-value for standing with usable ace")
    axes[0, 0].set_xlabel("Dealer showing")
    axes[0, 0].set_ylabel("Player sum")
    axes[0, 0].invert_yaxis()

    sns.heatmap(q_hit_ace, ax=axes[0, 1], xticklabels=new_labels_y, yticklabels=new_labels_x, vmin=vmin, vmax=vmax, cbar=False, annot=True, fmt=".2f")
    axes[0, 1].set_title("Q-value for hitting with usable ace")
    axes[0, 1].set_xlabel("Dealer showing")
    axes[0, 1].set_ylabel("Player sum")
    axes[0, 1].invert_yaxis()

    sns.heatmap(q_stand_no_ace, ax=axes[1, 0], xticklabels=new_labels_y, yticklabels=new_labels_x,  vmin=vmin, vmax=vmax, cbar=False, annot=True, fmt=".2f")
    axes[1, 0].set_title("Q-value for standing without usable ace")
    axes[1, 0].set_xlabel("Dealer showing")
    axes[1, 0].set_ylabel("Player sum")
    axes[1, 0].invert_yaxis()

    sns.heatmap(q_hit_no_ace, ax=axes[1, 1], xticklabels=new_labels_y, yticklabels=new_labels_x, vmin=vmin, vmax=vmax, cbar=False, annot=True, fmt=".2f")
    axes[1, 1].set_title("Q-value for hitting without usable ace")
    axes[1, 1].set_xlabel("Dealer showing")
    axes[1, 1].set_ylabel("Player sum")
    axes[1, 1].invert_yaxis()

    plt.tight_layout()
    plt.savefig("blackjack_q_values.png")
    print("done")



run()
env.close()
