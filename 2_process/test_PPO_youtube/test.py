# Import libraries
import gymnasium as gym
from stable_baselines3 import PPO

# Create training environment WITHOUT rendering (faster training)
env_train = gym.make('LunarLander-v3')

# Create and train the agent
model = PPO('MlpPolicy', env_train, verbose=1)
print("Training agent for 1000000 timesteps...")
model.learn(total_timesteps=1000000)
print("Training complete!")

# Create testing environment WITH rendering (visual feedback)
env_test = gym.make('LunarLander-v3', render_mode='human')

# Number of test episodes
episodes = 2

print(f"\nTesting trained agent for {episodes} episodes...\n")

# Test the trained agent
for ep in range(episodes):
    obs, info = env_test.reset()  # Properly unpack reset return values
    done = False
    total_reward = 0
    step_count = 0
    
    while not done:
        # Get predicted action from trained model
        action, _states = model.predict(obs)
        
        # Execute action and get all 5 return values (new Gymnasium API)
        obs, reward, terminated, truncated, info = env_test.step(action)
        
        total_reward += reward
        step_count += 1
        done = terminated or truncated
        
        env_test.render()
    
    print(f"Episode {ep + 1}: Steps={step_count}, Total Reward={total_reward:.2f}")

# Clean up
env_train.close()
env_test.close()
print("\nDone!")