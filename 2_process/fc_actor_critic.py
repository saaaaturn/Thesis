#//////////////////////////////////////////////////////////////////////////////////
#//Fully Connecter Layers for Actor-Critic Heads in PPO                          //
#//This file defines a simple FC actor-critic model                              //
#//Outputs at: - policy logits/probabilities for discrete actions                //
#//             - state value estimate                                           //
#//Usage: python fc_actor_critic.py --state-file state_320d.txt --num-actions 12 //
#//////////////////////////////////////////////////////////////////////////////////

#!/usr/bin/env python3
"""Stage 5 prototype: fully-connected actor-critic heads for PPO.

Consumes a fixed-size state vector (default 320D) and outputs:
- policy logits/probabilities over actions
- state value estimate

Usage:
  python fc_actor_critic.py --state-file state_320d.txt --num-actions 12
"""

#/////////////////////////////////////////////////////////////////////////////////
#/////////IMPORTS AND DEFINITIONS/////////////////////////////////////////////////
#/////////////////////////////////////////////////////////////////////////////////

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn

#////////////////////////////////////////////////////////////////////////////////
#/////////FC ACTOR-CRITIC MODEL DEFINITION///////////////////////////////////////
#////////////////////////////////////////////////////////////////////////////////

#Function to load state vector 
def load_state_vector(path: Path) -> torch.Tensor:
    """Load a 1D state vector from a whitespace-separated text file."""
    raw = path.read_text(encoding="utf-8").strip().split()
    if not raw:
        raise ValueError(f"Empty state file: {path}")
    vec = torch.tensor([float(x) for x in raw], dtype=torch.float32)
    return vec

# Fully-connected actor-critic model with shared trunk and separate heads for policy and value estimation.
class FCActorCritic(nn.Module):
    """Shared FC trunk with separate actor and critic heads."""

    def __init__(
        self,
        state_dim: int = 320,
        num_actions: int = 12,
        trunk_hidden: Tuple[int, int] = (256, 128),
        head_hidden: int = 64,
    ) -> None:
        super().__init__()

        h1, h2 = trunk_hidden
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
        )

        self.actor = nn.Sequential(
            nn.Linear(h2, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, num_actions),
        )

        self.critic = nn.Sequential(
            nn.Linear(h2, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if state.dim() == 1:
            state = state.unsqueeze(0)  # [1, D]

        latent = self.trunk(state)
        logits = self.actor(latent)              # [B, A]
        probs = torch.softmax(logits, dim=-1)    # [B, A]
        value = self.critic(latent)              # [B, 1]
        return logits, probs, value

#////////////////////////////////////////////////////////////////////////////////
#////////////////////////////////////////MAIN FUNCTION///////////////////////////
#////////////////////////////////////////////////////////////////////////////////

def main() -> None:
    parser = argparse.ArgumentParser(description="Run FC actor-critic on a saved state vector")
    parser.add_argument("--state-file", type=str, default="state_320d.txt", help="Input state vector file")
    parser.add_argument("--state-dim", type=int, default=320, help="Expected state vector dimension")
    parser.add_argument("--num-actions", type=int, default=12, help="Discrete action space size")
    parser.add_argument("--save-dir", type=str, default="FC_results", help="Directory to save FC outputs")
    args = parser.parse_args()

    state_path = Path(args.state_file)
    if not state_path.exists():
        raise FileNotFoundError(f"Missing state file: {state_path}")

    state = load_state_vector(state_path)
    if state.numel() != args.state_dim:
        raise ValueError(
            f"State dimension mismatch: expected {args.state_dim}, got {state.numel()} from {state_path}"
        )

    model = FCActorCritic(state_dim=args.state_dim, num_actions=args.num_actions)
    model.eval()

    with torch.no_grad():
        logits, probs, value = model(state)

    # Remove batch dimension for saving/printing.
    logits_1d = logits.squeeze(0)
    probs_1d = probs.squeeze(0)
    value_1d = value.squeeze(0)

    print("=== FC Actor-Critic Forward Pass Complete ===")
    print(f"Input state shape: {tuple(state.shape)}")
    print(f"Policy logits shape: {tuple(logits_1d.shape)}")
    print(f"Policy probs shape: {tuple(probs_1d.shape)}")
    print(f"Value shape: {tuple(value_1d.shape)}")
    print(f"Sum of action probabilities: {probs_1d.sum().item():.6f}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    logits_path = save_dir / "policy_logits.txt"
    probs_path = save_dir / "policy_probs.txt"
    value_path = save_dir / "state_value.txt"

    logits_path.write_text(" ".join(f"{v:.6f}" for v in logits_1d.tolist()) + "\n", encoding="utf-8")
    probs_path.write_text(" ".join(f"{v:.6f}" for v in probs_1d.tolist()) + "\n", encoding="utf-8")
    value_path.write_text(" ".join(f"{v:.6f}" for v in value_1d.tolist()) + "\n", encoding="utf-8")

    print(f"Saved policy logits to: {logits_path}")
    print(f"Saved policy probabilities to: {probs_path}")
    print(f"Saved state value to: {value_path}")


if __name__ == "__main__":
    main()

#////////////////////////////////////////////////////////////////////////////////
#////////////////////////////////////////END OF FILE/////////////////////////////
#////////////////////////////////////////////////////////////////////////////////