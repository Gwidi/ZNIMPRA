import gymnasium as gym
import numpy as np

from controllers.random import RandomController
from controllers.PD import PDController

n_steps = 200
env = gym.make("Pendulum-v1", render_mode="human", g=9.81)
controller = PDController(env)

env.reset()
initial_state = np.array([0.1, 0.5])
env.unwrapped.state = initial_state
observation = env.unwrapped._get_obs()

episode_reward = 0.
for i in range(n_steps):
    action = controller.compute_control(observation=observation)
    observation, reward, terminated, truncated, info = env.step(action)
    episode_reward += reward
    env.render()
print(f"Episode reward: {episode_reward}")
