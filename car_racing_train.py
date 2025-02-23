'''Tran car racing for first 100 episodes'''

from car_racing_env import get_env
from car_racing_agent import Agent
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import count
import tqdm
import numpy as np
import pandas as pd

num_episodes = 100

env = get_env()
state, info = env.reset()

agent = Agent(state_shape=state.shape, n_actions=env.action_space.n)
# agent.buffer.buffer.loads('checkpoints')

def report(rewards, losses, agent):
    print(f"\nMean reward: {sum(rewards[-10:])/10}")
    print(f"Mean loss: {sum(losses[-10:])/10}")
    print(f"Epsilon: {agent.e_greedy_policy.epsilon_schedule.vs[-1]}")

def plot_report(rewards, losses, epsilons):
    df = pd.DataFrame({
        "Reward": rewards,
        "Loss": losses,
        # "Epsilon": epsilons
    })
    df.to_csv("racing_dqn_continuous.csv")
    fig, ax = plt.subplots()
    sns.lineplot(data=rewards, ax=ax, label="Reward", color="blue")
    sns.lineplot(data=losses, ax=ax, label="Loss", color="red")
    # sns.lineplot(data=epsilons, ax=ax, label="Epsilon", color="green")
    plt.tight_layout()
    plt.savefig("racing_dqn_continuous.png")
    plt.close()


all_rewards = []
all_losses = []

for i_episode in tqdm.tqdm(range(num_episodes)):

    s, info = env.reset()
    
    # s = torch.tensor(state, dtype=torch.float32, device="cuda").unsqueeze(0)

    episode_reward = 0
    episode_loss = 0

    for t in count():
        
        # Collect data
        a = agent.act(s)
        s_, r, done, truncated, _ = env.step(a)
        episode_reward += r
        agent.buffer.push(s, a, r, s_, done)
        s = s_
        done = done or truncated
        
        # Update agent Q estimates
        if t % 4 == 0:
            loss = agent.update()
            episode_loss += loss
        
        if done:
            all_rewards.append(episode_reward)
            all_losses.append(episode_loss)
            break
    
    if i_episode % 1 == 0:
        report(all_rewards, all_losses, agent)
    
    if i_episode % 10 == 0:
        agent.sync_target()
        agent.save_checkpoint()
        plot_report(all_rewards, all_losses, np.array(agent.e_greedy_policy.epsilon_schedule.vs))
    
plot_report(all_rewards, all_losses, np.array(agent.e_greedy_policy.epsilon_schedule.vs))  
agent.save_checkpoint()
print('Complete')
# torch.save(policy_net.state_dict(), "racing_dqn_continuous.pth")
