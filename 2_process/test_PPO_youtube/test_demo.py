# Demo: Show all values returned by each step
import gymnasium as gym
import numpy as np

# Create environment WITHOUT rendering (much faster)
env = gym.make('LunarLander-v3')

# Reset to get initial observation
obs, info = env.reset()
print("=" * 70)
print("INITIAL STATE (from env.reset())")
print("=" * 70)
print(f"Observation shape: {obs.shape}")
print(f"Observation values: {obs}")
print(f"  x_pos={obs[0]:.4f}, y_pos={obs[1]:.4f}")
print(f"  x_vel={obs[2]:.4f}, y_vel={obs[3]:.4f}")
print(f"  angle={obs[4]:.4f}, ang_vel={obs[5]:.4f}")
print(f"  left_leg={obs[6]:.0f}, right_leg={obs[7]:.0f}")
print(f"Info dict: {info}")

# Take a few random actions to show step outputs
print("\n" + "=" * 70)
print("EXAMPLE STEPS (taking random actions)")
print("=" * 70)

action_names = ["Do nothing", "Fire left", "Fire main", "Fire right"]

for step in range(5):
    # Sample a random action
    action = env.action_space.sample()
    print(f"\n--- STEP {step + 1} ---")
    print(f"Taking action: {action} ({action_names[action]})")
    
    # Execute the action and get all return values
    obs, reward, terminated, truncated, info = env.step(action)
    
    print(f"\nReturned values from env.step():")
    print(f"  obs (observation): {obs}")
    print(f"    → Position: x={obs[0]:.4f}, y={obs[1]:.4f}")
    print(f"    → Velocity: vx={obs[2]:.4f}, vy={obs[3]:.4f}")
    print(f"    → Rotation: angle={obs[4]:.4f}, ang_vel={obs[5]:.4f}")
    print(f"    → Contact: left={obs[6]:.0f}, right={obs[7]:.0f}")
    
    print(f"  reward: {reward:.4f}")
    print(f"    → Each step costs -0.5")
    print(f"    → Crashing = -100, Landing = +100")
    
    print(f"  terminated: {terminated} (goal reached?)")
    print(f"  truncated: {truncated} (time limit?)")
    print(f"  info: {info}")

env.close()
