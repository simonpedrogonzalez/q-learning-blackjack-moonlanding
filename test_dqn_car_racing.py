''' tests the agent in the car racing environment. It takes some time to load '''

from car_racing_env import get_env2
from car_racing_agent import Agent
from itertools import count
import tqdm

num_episodes = 5

env = get_env2()
state, info = env.reset()

agent = Agent(state_shape=state.shape, n_actions=env.action_space.n)
agent.load_checkpoint()

env = get_env2()

for i_episode in tqdm.tqdm(range(num_episodes)):

    s, info = env.reset()
    
    for t in count():
        print(f"Step: {t}")
        a = agent.act(s)
        s_, r, done, truncated, _ = env.step(a)
        s = s_
        env.render()
        
        done = done or truncated
        
        if done:
            break
