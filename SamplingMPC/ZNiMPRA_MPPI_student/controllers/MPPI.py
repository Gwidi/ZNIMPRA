from controllers.interface import Controller
import numpy as np
import gymnasium as gym

class MPPIController(Controller):
    def __init__(self, env_name, sigma=1., N=10, T=30, lambda_=0.1, min_u=-2.0, max_u=2.0):
        if env_name == "Pendulum-v1":
            self.env = gym.make(env_name, g=9.81)
        elif env_name == "Hopper-v5":
            self.env = gym.make(env_name)
        else:
            raise ValueError(f"Unsupported environment: {env_name}")
        self.actions = self.env.action_space.shape[0]
        self.env.reset()
        super().__init__(self.env)       
        self.sigma = sigma
        self.N = N # Number of trajectories
        self.T = T # Time horizon
        self.lambda_ = lambda_ # Temperature parameter
        self.min_u = min_u
        self.max_u = max_u
         
        self.u_star = np.zeros((self.T, self.actions))  # Initial control sequence
    
        self.env_name = env_name
        
    def compute_control(self, env) -> np.ndarray:
        self.u_star = np.roll(self.u_star, -1, axis=0) # Shift control sequence
        self.u_star[-1, :] = 0 # Set last control to zero
        u = np.tile(self.u_star[:, :, None], (1, 1, self.N)) # Replicate control sequence for all trajectories
        delta_u = np.zeros((self.T, self.actions, self.N)) # To store control perturbations
        S = np.zeros(self.N) # To store cumulative rewards
        omega = np.zeros(self.N) # To store weights
        
        for traj in range(self.N):
            if self.env_name == "Pendulum-v1":
                self.env.unwrapped.state = env.unwrapped.state
            elif self.env_name == "Hopper-v5":
                qpos = env.unwrapped.data.qpos.copy()
                qvel = env.unwrapped.data.qvel.copy()
                self.env.unwrapped.set_state(qpos, qvel)
            for t in range(self.T):
                noise = np.random.normal(0, self.sigma, size=self.actions) # Generate noise
                u[t, :, traj] += noise
                u[t, :, traj] = np.clip(u[t, :, traj], self.min_u, self.max_u)
                delta_u[t, :, traj] = u[t, :, traj] - self.u_star[t, :]
                action = u[t, :, traj]
                observation, reward, terminated, truncated, info = self.env.step(action)
                S[traj] += reward
            S[traj] /= self.T
        
            omega[traj] = np.exp(np.array(S[traj]) / self.lambda_) # Compute weights
        
        omega /= np.sum(omega) # Normalize weights

        self.u_star += np.sum(delta_u * omega, axis=2) # Update control sequence


        return self.u_star[0]