import gymnasium as gym
import numpy as np

from controllers.random import RandomController
from controllers.PD import PDController
from controllers.SwingUp import SwingUpController
from controllers.BasicSampling import BasicSamplingController, plot_heatmap
from controllers.MPPI import MPPIController

def task2_3_4():
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

def task5():
    '''Basic Sampling Controller Implementation'''
    n_steps = 200 
    env = gym.make("Pendulum-v1", render_mode="human", g=9.81)
    env.reset()
    env.unwrapped.state = np.array([3.1, 0.0])
    controller = BasicSamplingController(env_name="Pendulum-v1", n_samples=50, seq_length=20)
    episode_reward = 0.
    
    for i in range(n_steps):
        action = controller.compute_control(env)
        observation, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        env.render()
    print(f"Episode reward: {episode_reward}")

def task6():
    '''Heatmap of Basic Sampling Controller Performance'''
    n_rollouts_list = [1, 5, 10, 20, 50, 100]
    seq_length_list = [1, 5, 10, 20, 50, 100]
    plot_heatmap(env_name="Pendulum-v1", n_rollouts_list=n_rollouts_list, seq_length_list=seq_length_list)

def task7():
    n_steps = 200
    env = gym.make("Pendulum-v1", render_mode="human", g=9.81)
    env.reset()
    env.unwrapped.state = np.array([3.1, 0.0])
    controller = MPPIController(env_name="Pendulum-v1")
    for i in range(n_steps):
        action = controller.compute_control(env)
        observation, reward, terminated, truncated, info = env.step(action)
        env.render()

def task9():
    n_steps = 200
    env = gym.make("Hopper-v5", render_mode="human")
    env.reset()
    controller = MPPIController(env_name="Hopper-v5", N=15, T=50, sigma=1.0, min_u=-1.0, max_u=1.0)
    for i in range(n_steps):
        action = controller.compute_control(env)
        observation, reward, terminated, truncated, info = env.step(action)
        env.render()

if __name__ == '__main__':
    task9()
