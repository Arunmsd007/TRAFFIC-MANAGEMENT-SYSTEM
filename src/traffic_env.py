import gymnasium as gym
from gymnasium import spaces
import numpy as np

class TrafficEnv(gym.Env):
    """
    Simple 4-lane traffic environment for RL agent.
    State: vehicle counts per lane [A, B, C, D]
    Action: which lane gets green (0=A,1=B,2=C,3=D)
    Reward: negative average waiting time (minimize queues)
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super(TrafficEnv, self).__init__()

        # state: number of vehicles in each lane (0–50)
        self.observation_space = spaces.Box(low=0, high=50, shape=(4,), dtype=np.int32)

        # action: choose green lane (0–3)
        self.action_space = spaces.Discrete(4)

        self.state = np.zeros(4, dtype=np.int32)
        self.time = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.random.randint(0, 10, size=(4,))
        self.time = 0
        return self.state, {}

    def step(self, action):
        # vehicles in chosen lane get cleared
        cleared = min(self.state[action], 3)  # 3 vehicles pass per step
        self.state[action] -= cleared

        # waiting time = total vehicles still in queue
        wait_time = sum(self.state)

        # reward = negative waiting (minimize queue length)
        reward = -wait_time

        # random new arrivals (Poisson distributed)
        arrivals = np.random.poisson(2, size=(4,))
        self.state = np.minimum(self.state + arrivals, 50)

        self.time += 1
        done = self.time >= 100  # episode length = 100 steps

        return self.state, reward, done, False, {}
