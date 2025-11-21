import gymnasium as gym
import numpy as np

from controllers.random import RandomController
from controllers.PD import PDController
from controllers.SwingUp import SwingUpController

n_steps = 200
env = gym.make("Pendulum-v1", render_mode="human", g=9.81)
controller1 = SwingUpController(env)
controller2 = PDController(env)

env.reset()
initial_state = np.array([3.1, 0.0])
env.unwrapped.state = initial_state
observation = env.unwrapped._get_obs()
cos_theta, sin_theta, theta_dot = observation
theta = np.arctan2(sin_theta, cos_theta)

episode_reward = 0.
for i in range(n_steps):
    if abs(theta) < np.pi/4:
        action = controller2.compute_control(observation=observation)
        print(observation)
    else:
        action = controller1.compute_control(observation=observation)
    observation, reward, terminated, truncated, info = env.step(action)
    cos_theta, sin_theta, theta_dot = observation
    theta = np.arctan2(sin_theta, cos_theta)
    episode_reward += reward
    env.render()
print(f"Episode reward: {episode_reward}")
