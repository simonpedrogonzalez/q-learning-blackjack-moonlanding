'''Custom environment for CarRacing'''

import gymnasium as gym
import gymnasium.wrappers as gym_wrap

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        for _ in range(self._skip):
            state, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated:
                break
        return state, total_reward, terminated, truncated, info

def get_env():
    env = gym.make(
        "CarRacing-v3",
        render_mode=None,
        lap_complete_percent=0.95,
        domain_randomize=False,
        continuous=False
    )
    env = SkipFrame(env, skip=4)
    env = gym_wrap.GrayscaleObservation(env)
    env = gym_wrap.ResizeObservation(env, shape=(84, 84))
    env = gym_wrap.FrameStackObservation(env, 4)

    return env

def get_env2():
    # This is for testing
    env = gym.make(
        "CarRacing-v3",
        render_mode="human",
        lap_complete_percent=0.95,
        domain_randomize=False,
        continuous=False
    )
    env = SkipFrame(env, skip=4)
    env = gym_wrap.GrayscaleObservation(env)
    env = gym_wrap.ResizeObservation(env, shape=(84, 84))
    env = gym_wrap.FrameStackObservation(env, 4)

    return env
    