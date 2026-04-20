# Import libraries
import gymnasium as gym
from stable_baselines3 import PPO
import os

VER_NAME = "PPO_test"
model_dir = f"models/{VER_NAME}"
log_dir = f"logs/{VER_NAME}"

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Create training environment WITHOUT rendering (faster training)
env_train = gym.make('LunarLander-v3')
env_train.reset()

# Create and train the agent
model = PPO('MlpPolicy', env_train, verbose=1, tensorboard_log=log_dir)
print("Training agent for number of timesteps...")

TIMESTEPS = 10000

for i in range(1, 30):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=VER_NAME)
    model.save(f"{model_dir}/{TIMESTEPS * i}")


print("Training complete!")

env_train.close() 
print("\nDone!")
