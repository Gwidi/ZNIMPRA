from controllers.interface import Controller
import numpy as np

class SwingUpController(Controller):
    def __init__(self, env):
        super().__init__(env)
        self.J = self.env.unwrapped.m * self.env.unwrapped.l ** 2
    def compute_control(self, observation):
        # Pendulum-v1 observation format: [cos(theta), sin(theta), theta_dot]
        cos_theta, sin_theta, theta_dot = observation
        theta = np.arctan2(sin_theta, cos_theta)
        E = 0.5 * self.J * theta_dot**2 + self.env.unwrapped.m * self.env.unwrapped.g * self.env.unwrapped.l * (np.cos(theta) - 1)
        E_desired = self.env.unwrapped.m * self.env.unwrapped.g * self.env.unwrapped.l
        u_max = self.env.action_space.high[0]
        u = u_max * self.env.unwrapped.g * np.sign((E - E_desired) * theta_dot * np.cos(theta))
        u = np.clip(u, self.env.action_space.low[0], self.env.action_space.high[0])
        return np.array([u])