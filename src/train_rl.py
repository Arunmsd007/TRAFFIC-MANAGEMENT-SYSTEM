from stable_baselines3 import DQN
from traffic_env import TrafficEnv

def train():
    env = TrafficEnv()
    model = DQN("MlpPolicy", env, verbose=1, learning_rate=0.001, buffer_size=10000)
    model.learn(total_timesteps=50000)
    model.save("models/rl_dqn")
    print("✅ RL model trained and saved to models/rl_dqn.zip")

if __name__ == "__main__":
    train()
