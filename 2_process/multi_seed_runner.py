#!/usr/bin/env python3
"""Multi-seed training runner for HybridSYN with deterministic eval protocol.

Phase 2: Systematic training with reproducibility, held-out test circuits, and structured reporting.

Usage:
  python multi_seed_runner.py --num-seeds 3 --total-timesteps 10000 --eval-circuits adder div
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import numpy as np

def run_command(cmd: List[str], description: str) -> int:
    """Run shell command and return exit code."""
    print(f"\n{'='*70}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {description}")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*70}\n")
    return subprocess.call(cmd)

def setup_experiment_dir(base_dir: str, experiment_name: str) -> Path:
    """Create experiment directory structure."""
    exp_dir = Path(base_dir) / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir

def train_seed(
    seed: int,
    exp_dir: Path,
    total_timesteps: int,
    eval_circuits: List[str],
    enforce_clean_signal: bool = True,
) -> Dict:
    """Train one seed and return results."""
    
    seed_dir = exp_dir / f"seed_{seed:02d}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    
    log_dir = seed_dir / "logs"
    model_dir = seed_dir / "models"
    
    cmd = [
        "python", "train_hybridsyn_ppo.py",
        "--seed", str(seed),
        "--total-timesteps", str(total_timesteps),
        "--log-dir", str(log_dir),
        "--model-dir", str(model_dir),
        "--best-by", "lut6",
        "--enforce-clean-signal" if enforce_clean_signal else "",
        "--eval-circuits", " ".join(eval_circuits),
    ]
    cmd = [c for c in cmd if c]  # Remove empty strings
    
    exit_code = run_command(cmd, f"Training Seed {seed:02d}/{total_timesteps} timesteps")
    
    if exit_code != 0:
        print(f"ERROR: Seed {seed:02d} training failed with exit code {exit_code}")
        return {
            "seed": seed,
            "success": False,
            "error": f"Training failed with code {exit_code}",
        }
    
    # Parse best model results.
    best_lut6_meta = model_dir / "best_lut6" / "best_lut6_meta.json"
    if best_lut6_meta.exists():
        with open(best_lut6_meta) as f:
            meta = json.load(f)
        return {
            "seed": seed,
            "success": True,
            "best_lut6": meta.get("best_lut6", None),
            "best_lut6_step": meta.get("best_lut6_step", None),
            "model_path": str(model_dir / "best_lut6" / "best_lut6_model.zip"),
        }
    else:
        print(f"WARNING: Could not find best_lut6_meta.json for seed {seed:02d}")
        return {
            "seed": seed,
            "success": True,
            "best_lut6": None,
            "best_lut6_step": None,
            "model_path": None,
        }

def evaluate_seed(
    seed: int,
    exp_dir: Path,
    test_circuits: List[str],
) -> Dict:
    """Evaluate trained model on held-out test circuits."""
    
    seed_dir = exp_dir / f"seed_{seed:02d}"
    model_dir = seed_dir / "models"
    eval_results = seed_dir / "eval_results.json"
    
    best_model = model_dir / "best_lut6" / "best_lut6_model.zip"
    if not best_model.exists():
        print(f"WARNING: Best model not found for seed {seed:02d}")
        return {
            "seed": seed,
            "success": False,
            "error": "Model not found",
        }
    
    cmd = [
        "python", "run_trained_model.py",
        "--model-path", str(best_model),
        "--circuits", " ".join(test_circuits),
        "--num-episodes", "3",
        "--output-json", str(eval_results),
    ]
    
    exit_code = run_command(cmd, f"Evaluating Seed {seed:02d} on test circuits")
    
    if exit_code == 0 and eval_results.exists():
        with open(eval_results) as f:
            results = json.load(f)
        return {
            "seed": seed,
            "success": True,
            "eval_results": results,
        }
    else:
        return {
            "seed": seed,
            "success": False,
            "error": f"Evaluation failed with code {exit_code}",
        }

def aggregate_results(exp_dir: Path, num_seeds: int) -> Dict:
    """Aggregate results across all seeds."""
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "num_seeds": num_seeds,
        "seeds": [],
        "aggregated": {},
    }
    
    best_lut6_values = []
    
    for seed in range(num_seeds):
        seed_dir = exp_dir / f"seed_{seed:02d}"
        model_dir = seed_dir / "models"
        best_lut6_meta = model_dir / "best_lut6" / "best_lut6_meta.json"
        
        if best_lut6_meta.exists():
            with open(best_lut6_meta) as f:
                meta = json.load(f)
            lut6 = meta.get("best_lut6")
            if lut6 is not None:
                best_lut6_values.append(lut6)
                results["seeds"].append({
                    "seed": seed,
                    "best_lut6": lut6,
                    "step": meta.get("best_lut6_step"),
                })
    
    if best_lut6_values:
        results["aggregated"]["best_lut6_mean"] = float(np.mean(best_lut6_values))
        results["aggregated"]["best_lut6_std"] = float(np.std(best_lut6_values))
        results["aggregated"]["best_lut6_min"] = float(np.min(best_lut6_values))
        results["aggregated"]["best_lut6_max"] = float(np.max(best_lut6_values))
    
    return results

def main():
    parser = argparse.ArgumentParser(
        description="Multi-seed training runner for HybridSYN Phase 2"
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=3,
        help="Number of seeds to run (default: 3)",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=10000,
        help="Total timesteps per seed (default: 10000)",
    )
    parser.add_argument(
        "--eval-circuits",
        nargs="+",
        default=["adder"],
        help="Circuits to evaluate during training (default: adder)",
    )
    parser.add_argument(
        "--test-circuits",
        nargs="+",
        default=["div", "hyp"],
        help="Held-out test circuits for final evaluation (default: div hyp)",
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        default=None,
        help="Experiment name (default: phase2_<timestamp>)",
    )
    parser.add_argument(
        "--exp-dir",
        type=str,
        default="experiments",
        help="Base directory for experiments (default: experiments)",
    )
    parser.add_argument(
        "--enforce-clean-signal",
        action="store_true",
        default=True,
        help="Enforce clean signal (default: True)",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip final evaluation on test circuits",
    )
    
    args = parser.parse_args()
    
    # Setup experiment.
    if args.exp_name is None:
        args.exp_name = f"phase2_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    exp_dir = setup_experiment_dir(args.exp_dir, args.exp_name)
    print(f"\nPhase 2 Multi-Seed Experiment: {args.exp_name}")
    print(f"Experiment directory: {exp_dir}")
    print(f"Number of seeds: {args.num_seeds}")
    print(f"Timesteps per seed: {args.total_timesteps}")
    print(f"Eval circuits (train): {args.eval_circuits}")
    print(f"Test circuits (held-out): {args.test_circuits}")
    print()
    
    # Train all seeds.
    print(f"\n{'='*70}")
    print("PHASE 2: TRAINING")
    print(f"{'='*70}")
    
    for seed in range(args.num_seeds):
        train_seed(
            seed=seed,
            exp_dir=exp_dir,
            total_timesteps=args.total_timesteps,
            eval_circuits=args.eval_circuits,
            enforce_clean_signal=args.enforce_clean_signal,
        )
    
    # Aggregate training results.
    print(f"\n{'='*70}")
    print("PHASE 2: AGGREGATING RESULTS")
    print(f"{'='*70}")
    
    agg_results = aggregate_results(exp_dir, args.num_seeds)
    
    print(f"\nAggregated Results (Validation on {args.eval_circuits}):")
    print(f"  Best LUT6 - Mean: {agg_results['aggregated'].get('best_lut6_mean', 'N/A'):.2f}")
    print(f"  Best LUT6 - Std:  {agg_results['aggregated'].get('best_lut6_std', 'N/A'):.2f}")
    print(f"  Best LUT6 - Min:  {agg_results['aggregated'].get('best_lut6_min', 'N/A'):.2f}")
    print(f"  Best LUT6 - Max:  {agg_results['aggregated'].get('best_lut6_max', 'N/A'):.2f}")
    
    # Save aggregated results.
    results_file = exp_dir / "aggregated_results.json"
    with open(results_file, "w") as f:
        json.dump(agg_results, f, indent=2)
    print(f"\nResults saved to: {results_file}")
    
    # Optional: Evaluate on held-out test circuits.
    if not args.skip_eval:
        print(f"\n{'='*70}")
        print("PHASE 2: EVALUATION ON HELD-OUT TEST CIRCUITS")
        print(f"{'='*70}")
        
        for seed in range(args.num_seeds):
            evaluate_seed(
                seed=seed,
                exp_dir=exp_dir,
                test_circuits=args.test_circuits,
            )
    
    print(f"\n{'='*70}")
    print("Phase 2 Multi-Seed Run Complete")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
