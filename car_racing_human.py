import gymnasium as gym
import pygame
import numpy as np


env = gym.make(
    "CarRacing-v3",
    render_mode="human",
    lap_complete_percent=0.95,
    domain_randomize=False,
    continuous=True
)

pygame.init()
screen = pygame.display.set_mode((400, 300))
pygame.display.set_caption("Car Racing - Human Input")


def get_human_action():
    """Captures keyboard inputs and returns a NumPy array for continuous control."""
    pygame.event.pump()
    keys = pygame.key.get_pressed()
    
    steering = 0.0
    acceleration = 0.0
    braking = 0.0

    # Steering control
    if keys[pygame.K_LEFT]:
        steering = -1.0
    elif keys[pygame.K_RIGHT]:
        steering = 1.0

    # Acceleration and braking
    if keys[pygame.K_UP]:
        acceleration = 1.0
    if keys[pygame.K_DOWN]:
        braking = 1.0

    return np.array([steering, acceleration, braking], dtype=np.float32)


obs, info = env.reset()
done = False

while not done:
    action = get_human_action()
    obs, reward, terminated, truncated, _ = env.step(action)  
    env.render()

    done = terminated or truncated

pygame.quit()
env.close()
