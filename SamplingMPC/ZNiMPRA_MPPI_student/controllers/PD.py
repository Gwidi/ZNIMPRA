from controllers.interface import Controller
import numpy as np

class PDController(Controller):
    def compute_control(self, observation, kp=10.0, kd=1.0):
        # Pendulum-v1 observation format: [cos(theta), sin(theta), theta_dot]
        cos_theta, sin_theta, theta_dot = observation
        theta = np.arctan2(sin_theta, cos_theta)
        desired_theta, desired_theta_dot = 0.0, 0.0 # Upright position
        # Desired position is upright (theta = 0)
        error = desired_theta - theta
        error_dot = desired_theta_dot - theta_dot
        # PD control law
        u = kp * error + kd * error_dot
        # Clip action to be within action space
        u = np.clip(u, self.env.action_space.low[0], self.env.action_space.high[0])
        return np.array([u])