from stable_baselines3 import DQN
from traffic_env import TrafficEnv
import numpy as np

def evaluate(model, episodes=50):
    env = TrafficEnv()
    rewards = []
    for _ in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            action, _ = model.predict(state, deterministic=True)
            state, reward, done, _, _ = env.step(action)
            total_reward += reward
        rewards.append(total_reward)
    return np.mean(rewards)

if __name__ == "__main__":
    model = DQN.load("models/rl_dqn")
    avg_reward = evaluate(model)
    print(f"✅ Average reward over 50 episodes: {avg_reward}")
