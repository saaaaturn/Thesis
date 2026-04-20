import os
import sys
from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
import matplotlib.pyplot as plt

# Use absolute path to logs directory
log_base = '/home/nam-bui/WORK/THESIS/TN/HybridSYN/logs/PPO_test/PPO_test_0'

print("=" * 80)
print("TENSORBOARD LOG VIEWER")
print("=" * 80)
print(f"Looking for logs at: {log_base}")

if not os.path.exists(log_base):
    print(f"✗ Error: Directory not found: {log_base}")
    print("\nAvailable directories:")
    root = '/home/nam-bui/WORK/THESIS/TN/HybridSYN/logs'
    if os.path.exists(root):
        for d in os.walk(root):
            print(f"  {d[0]}")
    sys.exit(1)

# Load TensorBoard events
print("✓ Directory found!")
ea = event_accumulator.EventAccumulator(log_base)
ea.Reload()

# Get available metrics
available_tags = ea.Tags()['scalars']
print(f"\nAvailable metrics ({len(available_tags)} total):")
for i, tag in enumerate(available_tags, 1):
    print(f"  {i}. {tag}")

# Let user choose which metric to view
print("\n" + "=" * 80)
choice = input("Enter metric name (or number, or just press Enter for 'rollout/ep_rew_mean'): ").strip()

if not choice:
    tag = 'rollout/ep_rew_mean'
elif choice.isdigit():
    idx = int(choice) - 1
    if 0 <= idx < len(available_tags):
        tag = available_tags[idx]
    else:
        print(f"Invalid choice. Using 'rollout/ep_rew_mean'")
        tag = 'rollout/ep_rew_mean'
else:
    tag = choice

if tag not in available_tags:
    print(f"Metric '{tag}' not found!")
    sys.exit(1)

# Extract the selected metric
print(f"\nLoading: {tag}")
events = ea.Scalars(tag)
steps = [e.step for e in events]
values = [e.value for e in events]

df = pd.DataFrame({'step': steps, 'value': values})

# Display data
print("\n" + "=" * 80)
print(f"METRIC: {tag}")
print("=" * 80)
print(f"Total data points: {len(df)}")
print(f"\nLast 10 values:")
print(df.tail(10).to_string(index=False))

print(f"\nStatistics:")
print(f"  Mean:   {df['value'].mean():.4f}")
print(f"  Max:    {df['value'].max():.4f}")
print(f"  Min:    {df['value'].min():.4f}")
print(f"  Last:   {df['value'].iloc[-1]:.4f}")

# Plot
print("\nGenerating plot...")
plt.figure(figsize=(12, 6))
plt.plot(df['step'], df['value'], linewidth=2, marker='o', markersize=4)
plt.xlabel('Training Step', fontsize=12)
plt.ylabel(tag, fontsize=12)
plt.title(f'{tag} Over Training', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'log_{tag.replace("/", "_")}.png', dpi=100)
print(f"✓ Saved to: log_{tag.replace('/', '_')}.png")
plt.show()