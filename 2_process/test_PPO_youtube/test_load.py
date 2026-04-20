# Import libraries
import gymnasium as gym
from stable_baselines3 import PPO

VER_NAME = "PPO_test"

# Create training environment WITHOUT rendering (faster training)
env_train = gym.make('LunarLander-v3', render_mode='human')
env_train.reset()

model_dir = f"models/{VER_NAME}"
log_dir = f"logs/{VER_NAME}"
model_path = f"{model_dir}/180000"

print(f"Loading model from {model_path}...")
model = PPO.load(model_path, env=env_train)

EPISODES = 5

print(f"Testing agent for {EPISODES} episodes...")
for ep in range(EPISODES):
    obs, info = env_train.reset()
    done = False
    total_reward = 0

    while not done:
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env_train.step(action)
        total_reward += reward
        done = terminated or truncated

    print(f"Episode {ep + 1}: Total Reward: {total_reward}")

print("Training complete!")

env_train.close() 
print("\nDone!")
