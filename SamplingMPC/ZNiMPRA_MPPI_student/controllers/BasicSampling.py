# This code is based on 
from controllers.interface import Controller
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

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
        
        self.env.unwrapped.state = env.unwrapped.state.copy()

        for _ in range(self.n_samples):
            state = self.env.unwrapped.state
            total_reward = 0.
            actions = []

            for _ in range(self.seq_length):
                action = self.env.action_space.sample()
                actions.append(action)
                observation, reward, terminated, truncated, info = self.env.step(action)
                total_reward += reward
                if terminated or truncated:
                    break

            if total_reward > max_reward:
                max_reward = total_reward
                best_action = actions[0]

        return np.array([best_action])