# This code is based on 
from controllers.interface import Controller
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import seaborn as sns

class BasicSamplingController(Controller):
    def __init__(self, env_name="Pendulum-v1", n_samples: int = 100, seq_length: int = 10):
        self.env = gym.make(env_name, g=9.81)
        self.env.reset(seed=42)
        super().__init__(self.env)
        self.n_samples = n_samples
        self.seq_length = seq_length

    def compute_control(self, env) -> np.ndarray:
        best_action = None
        max_reward = -np.inf
        
        self.env.unwrapped.state = env.unwrapped.state

        for _ in range(self.n_samples):
            state = self.env.unwrapped.state
            total_reward = 0.
            actions = []

            for _ in range(self.seq_length):
                action = self.env.action_space.sample()
                actions.append(action)
                observation, reward, terminated, truncated, info = self.env.step(action)
                total_reward += reward

            self.env.unwrapped.state = state

            if total_reward > max_reward:
                max_reward = total_reward
                best_action = actions[0]

        return np.array([best_action])
    
def plot_heatmap(env_name, n_rollouts_list, seq_length_list):
    rewards = np.zeros((len(n_rollouts_list), len(seq_length_list)))
    
    for i, n_rollouts in enumerate(n_rollouts_list):
        for j, seq_length in enumerate(seq_length_list):
            env = gym.make(env_name, g=9.81)
            env.reset(seed=42)
            env.unwrapped.state = np.array([np.pi, 0.0])
            controller = BasicSamplingController(env_name=env_name, n_samples=n_rollouts, seq_length=seq_length)
            episode_reward = 0.
            n_steps = 200
            
            for _ in range(n_steps):
                action = controller.compute_control(env)
                observation, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                if terminated or truncated:
                    break
            
            rewards[i, j] = episode_reward
            env.close()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(rewards, xticklabels=seq_length_list, yticklabels=n_rollouts_list, annot=True, fmt=".1f")
    plt.xlabel('Sequence Length')
    plt.ylabel('Number of Rollouts')
    plt.title('Heatmap of Episode Rewards')
    plt.show()