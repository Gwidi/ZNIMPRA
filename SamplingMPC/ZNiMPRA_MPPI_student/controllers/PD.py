from controllers.interface import Controller
import numpy as np

class PDController(Controller):
    def compute_control(self, initial_state, kp=10.0, kd=2.0):
        observation = initial_state
        desired_state = np.array([0.0, 0.0])
        # Desired position is upright (theta = 0)
        error = desired_state[0] - observation[0]
        error_dot = desired_state[1] - observation[1]
        # PD control law
        u = kp * error + kd * error_dot
        # Clip action to be within action space
        u = np.clip(u, self.env.action_space.low[0], self.env.action_space.high[0])
        return np.array([u])