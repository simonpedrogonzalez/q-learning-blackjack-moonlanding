'''
Human and Random Policy Comparison for Lunar Lander
'''

import gymnasium as gym
import argparse
import pygame
import seaborn as sns
import matplotlib.pyplot as plt
import tqdm


def simple_run(teleop=False):

    env = gym.make("LunarLander-v3", continuous=False, gravity=-10.0,
                   enable_wind=False, wind_power=15.0, turbulence_power=1.5, render_mode="human")
    obs, info = env.reset()

    done = False
    rewards = []
    while not done:
        # Control Branch for human teleop (Teleop=True, Heuristic=False)
        action = 0  # Default action = do nothing
        if teleop:
            pygame.event.pump()
            keys = pygame.key.get_pressed()

            if keys[pygame.K_LEFT]:
                action = 1  # Fire left orientation engine
            elif keys[pygame.K_UP]:
                action = 2  # Fire main engine
            elif keys[pygame.K_RIGHT]:
                action = 3  # Fire right orientation engine
            else:
                action = 0  # Default to doing nothing

        # Control Branch for Random Actions (Teleop=False, Heuristic=False)
        else:
            # Randomly Sample an Action
            action = env.action_space.sample()

        obs, rew, done, truncated, info = env.step(action)
        # print("ACt:", action)
        rewards.append(rew)
        env.render()

    print("Cumulative Reward:", sum(rewards))
    return sum(rewards)

def random_policy():
    env = gym.make("LunarLander-v3", continuous=False, gravity=-10.0,
                   enable_wind=False, wind_power=15.0, turbulence_power=1.5, render_mode=None)
    obs, info = env.reset()

    n_rounds  = 400
    c_rewards = []
    for i in tqdm.tqdm(range(n_rounds)):
        done = False
        rewards = []
        while not done:
            action = env.action_space.sample()
            obs, rew, done, truncated, info = env.step(action)
            env.render()
            rewards.append(rew)
        c_rewards.append(sum(rewards))
        obs, info = env.reset()
        done = False
    env.close()
    print("Average Reward By Random Policy:", sum(c_rewards)/n_rounds)
    return c_rewards

if __name__ == "__main__":
    params = argparse.ArgumentParser()
    params.add_argument("--teleop", action="store_true", default=False, help="Use the keyboard keys to control the movement")
    #If you don't specify --teleop it will default to a random policy

    args = params.parse_args()
    rs = []
    for i in range(20):
        r = simple_run(args.teleop)
        rs.append(r)
    print("Average Reward By Human:", sum(rs)/len(rs))
    
    
    human_rs = rs
    random_rewards = random_policy()

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    sns.lineplot(data=human_rs, ax=ax[0])
    ax[0].axhline(sum(human_rs)/len(human_rs), color='orange', linestyle='--')
    ax[0].set_xlabel("Episode")
    ax[0].set_ylabel("Reward")
    ax[0].set_title("Human Control")

    sns.lineplot(data=random_rewards, ax=ax[1])
    ax[1].axhline(sum(random_rewards)/len(random_rewards), color='orange', linestyle='--')
    ax[1].set_xlabel("Episode")
    ax[1].set_ylabel("Reward")
    ax[1].set_title("Random Policy")
    plt.tight_layout()
    plt.savefig("lunar_lander_comparison.png")
    print('done')

