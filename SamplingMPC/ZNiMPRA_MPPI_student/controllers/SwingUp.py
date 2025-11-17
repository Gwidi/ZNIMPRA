from controllers.interface import Controller
import numpy as np

class SwingUpController(Controller):
    def compute_control(self):
        E = 1/2 * self.env.unwrapped.J * self.env.state[1]**2 + self.env.unwrapped.m * self.env.unwrapped.g * self.env.unwrapped.l * (np.cos(self.env.state[0]) - 1)
        E_desired = 0.0
        u_max = self.env.action_space.high[0]
        u = u_max * np.sign(E - E_desired)* self.env.state[1] * np.cos(self.env.state[0])
        u = np.clip(u, self.env.action_space.low[0], self.env.action_space.high[0])
        return np.array([u])