import gymnasium as gym

from controllers.random import RandomController

n_steps = 200
env = gym.make("Pendulum-v1", render_mode="human", g=9.81)
controller = RandomController(env)

env.reset()

episode_reward = 0.
for i in range(n_steps):
    action = controller.compute_control()
    next_observation, reward, terminated, truncated, info = env.step(action)
    episode_reward += reward
    env.render()
print(f"Episode reward: {episode_reward}")
