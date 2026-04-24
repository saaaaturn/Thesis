#!/usr/bin/env python3
"""Run a trained PPO model for inference episodes and save detailed results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import sys
from typing import Any, TextIO

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from hybridsyn_env import HybridSYNEnv


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CIRCUIT_FILE = "../1_inputs/EPFL_benchmarks/arithmetic/adder.aig"


class TeeStream:
    """Mirror writes to both terminal and a log file."""

    def __init__(self, *streams: TextIO) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return (SCRIPT_DIR / path).resolve()


def infer_run_name(model_path: Path) -> str:
    if model_path.parent.name == "best":
        return model_path.parent.parent.name
    return model_path.parent.name


def infer_run_dir(model_path: Path) -> Path:
    if model_path.parent.name in {"best", "checkpoints"}:
        return model_path.parent.parent
    return model_path.parent


def infer_model_name(model_path: Path) -> str:
    """Infer a stable model-name folder from the selected model artifact."""
    if model_path.suffix:
        return model_path.stem
    return model_path.name


def to_json_safe(value: Any) -> Any:
    """Convert nested objects into JSON-serializable primitives."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_safe(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return str(value)


def to_float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def load_run_args(model_path: Path) -> dict[str, Any]:
    run_dir = infer_run_dir(model_path)
    run_args_path = run_dir / "run_args.json"
    if not run_args_path.exists():
        return {}
    try:
        return json.loads(run_args_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a trained PPO model on HybridSYN env")
    parser.add_argument("--model-path", type=str, required=True, help="Path to model zip (best/final/checkpoint)")
    parser.add_argument("--mode", type=str, choices=["auto", "single", "multiple"], default="auto")
    parser.add_argument("--circuit-file", type=str, default=DEFAULT_CIRCUIT_FILE)
    parser.add_argument("--circuit-pool", nargs="+", default=[])
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--sequence-steps", type=int, default=25)
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic policy")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--matrix-run-tag", type=str, default="", help="Override tag for matrix/320D outputs")
    parser.add_argument("--output-dir", type=str, default="", help="Directory to save inference logs and trace")
    parser.add_argument("--reward-alpha", type=float, default=None, help="Override reward alpha (LUT6 weight); defaults to training run value (1.0)")
    parser.add_argument("--reward-beta", type=float, default=None, help="Override reward beta (Level weight); defaults to training run value (0.0)")
    parser.add_argument("--step-penalty", type=float, default=None, help="Override step penalty; defaults to training run value (0.0)")
    parser.add_argument("--reward-scale", type=float, default=None, help="Override reward scale; defaults to training run value")
    parser.add_argument("--reward-clip", type=float, default=None, help="Override reward clip; defaults to training run value")
    parser.add_argument("--allowed-actions", nargs="+", default=[])
    parser.add_argument(
        "--enforce-clean-signal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Terminate sequence and penalize when action execution/re-encode pipeline fails.",
    )
    return parser.parse_args()


def infer_circuit_label(mode: str, circuit_file: Path, circuit_pool: list[Path]) -> str:
    if mode == "multiple":
        if len(circuit_pool) > 1:
            return f"pool_{len(circuit_pool)}"
        if circuit_pool:
            return circuit_pool[0].stem
    return circuit_file.stem


def main() -> None:
    args = parse_args()

    model_path = resolve_path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")

    run_args = load_run_args(model_path)

    if args.mode == "auto":
        mode = str(run_args.get("mode", "single"))
    else:
        mode = args.mode

    reward_alpha = args.reward_alpha if args.reward_alpha is not None else float(run_args.get("reward_alpha", 1.0))
    reward_beta = args.reward_beta if args.reward_beta is not None else float(run_args.get("reward_beta", 0.0))
    step_penalty = args.step_penalty if args.step_penalty is not None else float(run_args.get("step_penalty", 0.0))
    reward_scale = args.reward_scale if args.reward_scale is not None else float(run_args.get("reward_scale", 10.0))
    reward_clip = args.reward_clip if args.reward_clip is not None else float(run_args.get("reward_clip", 10.0))
    allowed_actions = args.allowed_actions or run_args.get("allowed_actions", []) or []

    if mode == "multiple":
        pool_values = args.circuit_pool or run_args.get("circuit_pool", []) or []
        if not pool_values:
            raise ValueError("multiple mode requires --circuit-pool or run_args.json with circuit_pool")
        circuit_pool = [resolve_path(str(item)) for item in pool_values]
        circuit_file = circuit_pool[0]
    else:
        circuit_pool = []
        if args.mode == "auto" and args.circuit_file == DEFAULT_CIRCUIT_FILE:
            circuit_file = resolve_path(str(run_args.get("circuit_file", DEFAULT_CIRCUIT_FILE)))
        else:
            circuit_file = resolve_path(args.circuit_file)

    circuit_name = infer_circuit_label(mode, circuit_file, circuit_pool)
    run_name = infer_run_name(model_path)
    model_name = infer_model_name(model_path)

    output_dir = (
        resolve_path(args.output_dir)
        if args.output_dir
        else (
            SCRIPT_DIR
            / "../3_outputs"
            / "after_training"
            / ("multiple" if mode == "multiple" else "single")
            / circuit_name
            / run_name
            / model_name
            / "inference"
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "inference_run.log"

    matrix_run_tag = args.matrix_run_tag or f"inference_{run_name}"

    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    log_file = log_path.open("w", encoding="utf-8")
    sys.stdout = TeeStream(orig_stdout, log_file)
    sys.stderr = TeeStream(orig_stderr, log_file)

    env = None
    try:
        env = Monitor(
            HybridSYNEnv(
                circuit_file=str(circuit_file),
                mode=mode,
                circuit_pool_files=[str(p) for p in circuit_pool],
                matrix_run_tag=matrix_run_tag,
                max_steps=args.sequence_steps,
                seed=args.seed,
                reward_alpha=reward_alpha,
                reward_beta=reward_beta,
                step_penalty=step_penalty,
                reward_scale=reward_scale,
                reward_clip=reward_clip,
                allowed_actions=allowed_actions,
                enforce_clean_signal=args.enforce_clean_signal,
            )
        )
        model = PPO.load(str(model_path), env=env, device=args.device)

        print("=== Inference run begin ===")
        print(f"Model: {model_path}")
        print(f"Mode: {mode}")
        print(f"Circuit: {circuit_file}")
        if circuit_pool:
            print(f"Circuit pool: {[str(p) for p in circuit_pool]}")
        print(f"Episodes: {args.episodes}")
        print(f"Deterministic: {args.deterministic}")
        print(f"Output dir: {output_dir}")
        print(f"Run log: {log_path}")
        print(f"Matrix run tag: {matrix_run_tag}")
        print(
            f"Reward params: alpha={reward_alpha} beta={reward_beta} "
            f"step_penalty={step_penalty} scale={reward_scale} clip={reward_clip}"
        )

        all_episodes: list[dict[str, Any]] = []
        csv_rows: list[dict[str, Any]] = []

        for ep in range(1, args.episodes + 1):
            obs, reset_info = env.reset(seed=args.seed + ep)
            done = False
            total_reward = 0.0
            episode_steps: list[dict[str, Any]] = []
            final_info: dict[str, Any] | None = None

            initial_metrics_raw = reset_info.get("metrics") or {}
            initial_qor = to_float_or_none(initial_metrics_raw.get("qor_score"))
            initial_and = initial_metrics_raw.get("and_count")
            initial_inv = initial_metrics_raw.get("inv_count")
            initial_lev = initial_metrics_raw.get("lev")
            initial_lut6 = initial_metrics_raw.get("lut6_count")
            initial_lut6_lev = initial_metrics_raw.get("lut6_lev")
            initial_state_file = reset_info.get("state_file")
            initial_state_name = Path(str(initial_state_file)).name if initial_state_file else "n/a"
            print(
                f"[Episode {ep:03d}] before model run | "
                f"Initial QoR={initial_qor} AND={initial_and} INV={initial_inv} LEV={initial_lev} "
                f"LUT6={initial_lut6} LUT6_Level={initial_lut6_lev} "
                f"Initial 320D={initial_state_name}"
            )

            while not done:
                action, _ = model.predict(obs, deterministic=args.deterministic)
                obs, reward, terminated, truncated, info = env.step(int(action))
                done = bool(terminated or truncated)
                total_reward += float(reward)
                final_info = info

                step_entry = {
                    "step": int(info.get("step", 0)),
                    "action_id": int(info.get("action_id", -1)),
                    "action_name": str(info.get("action_name", "n/a")),
                    "reward": float(reward),
                    "qor_prev": to_float_or_none(info.get("qor_prev")),
                    "qor_next": to_float_or_none(info.get("qor_next")),
                    "state_file": info.get("state_file"),
                    "circuit": info.get("circuit"),
                    "metrics_prev": to_json_safe(info.get("metrics_prev")),
                    "metrics_next": to_json_safe(info.get("metrics_next")),
                    "lut6_count": to_float_or_none((info.get("metrics_next") or {}).get("lut6_count")),
                    "lut6_lev": to_float_or_none((info.get("metrics_next") or {}).get("lut6_lev")),
                }
                episode_steps.append(step_entry)
                csv_rows.append(
                    {
                        "episode": ep,
                        "step": step_entry["step"],
                        "action_id": step_entry["action_id"],
                        "action_name": step_entry["action_name"],
                        "reward": step_entry["reward"],
                        "qor_prev": step_entry["qor_prev"],
                        "qor_next": step_entry["qor_next"],
                        "state_file": step_entry["state_file"],
                        "circuit": step_entry["circuit"],
                        "lut6_count": step_entry["lut6_count"],
                        "lut6_lev": step_entry["lut6_lev"],
                    }
                )

                metrics_next_raw = info.get("metrics_next") or {}
                and_count = metrics_next_raw.get("and_count", "n/a")
                inv_count = metrics_next_raw.get("inv_count", "n/a")
                depth = metrics_next_raw.get("lev", "n/a")
                lut6_count = metrics_next_raw.get("lut6_count", "n/a")
                lut6_lev = metrics_next_raw.get("lut6_lev", "n/a")
                print(
                    f"[Episode {ep:03d}] [Step {step_entry['step']:03d}] "
                    f"Action={step_entry['action_name']} "
                    f"Current QoR={step_entry['qor_next']} "
                    f"Reward={step_entry['reward']:+.6f} "
                    f"Area(AND={and_count}, INV={inv_count}) "
                    f"Depth={depth} "
                    f"LUT6={lut6_count} LUT6_Level={lut6_lev}"
                )

            initial_metrics = to_json_safe(initial_metrics_raw)
            final_metrics = to_json_safe((final_info or {}).get("metrics_next") if final_info else None)
            final_qor = to_float_or_none((final_info or {}).get("qor_next"))
            qor_delta = None
            if initial_qor is not None and final_qor is not None:
                qor_delta = final_qor - initial_qor

            episode_summary = {
                "episode": ep,
                "steps": len(episode_steps),
                "total_reward": float(total_reward),
                "initial_qor": initial_qor,
                "final_qor": final_qor,
                "qor_delta": qor_delta,
                "initial_state_file": reset_info.get("state_file"),
                "final_state_file": (final_info or {}).get("state_file"),
                "initial_circuit": reset_info.get("circuit"),
                "final_circuit": (final_info or {}).get("circuit"),
                "initial_metrics": initial_metrics,
                "final_metrics": final_metrics,
                "trace": episode_steps,
            }
            all_episodes.append(episode_summary)
            print(
                f"[Episode {ep:03d}] end | steps={episode_summary['steps']} "
                f"total_reward={episode_summary['total_reward']:+.6f} "
                f"initial_qor={episode_summary['initial_qor']} "
                f"final_qor={episode_summary['final_qor']} "
                f"qor_delta={episode_summary['qor_delta']}"
            )

        total_rewards = [float(ep["total_reward"]) for ep in all_episodes]
        final_qors = [to_float_or_none(ep["final_qor"]) for ep in all_episodes]
        valid_final_qors = [v for v in final_qors if v is not None]
        steps_list = [int(ep["steps"]) for ep in all_episodes]

        aggregate = {
            "episodes": args.episodes,
            "total_steps": sum(steps_list),
            "mean_episode_steps": statistics.fmean(steps_list) if steps_list else 0.0,
            "mean_total_reward": statistics.fmean(total_rewards) if total_rewards else 0.0,
            "std_total_reward": statistics.pstdev(total_rewards) if len(total_rewards) > 1 else 0.0,
            "min_total_reward": min(total_rewards) if total_rewards else None,
            "max_total_reward": max(total_rewards) if total_rewards else None,
            "mean_final_qor": statistics.fmean(valid_final_qors) if valid_final_qors else None,
            "std_final_qor": statistics.pstdev(valid_final_qors) if len(valid_final_qors) > 1 else 0.0,
            "min_final_qor": min(valid_final_qors) if valid_final_qors else None,
            "max_final_qor": max(valid_final_qors) if valid_final_qors else None,
            "best_episode_by_reward": max(all_episodes, key=lambda x: x["total_reward"])["episode"] if all_episodes else None,
            "best_episode_by_final_qor": (
                max((ep for ep in all_episodes if ep["final_qor"] is not None), key=lambda x: x["final_qor"])["episode"]
                if any(ep["final_qor"] is not None for ep in all_episodes)
                else None
            ),
        }

        trace_path = output_dir / "inference_trace.json"
        summary_path = output_dir / "inference_summary.json"
        csv_path = output_dir / "inference_steps.csv"

        trace_path.write_text(json.dumps(to_json_safe(all_episodes), indent=2), encoding="utf-8")
        summary_payload = {
            "model_path": str(model_path),
            "circuit_file": str(circuit_file),
            "deterministic": args.deterministic,
            "matrix_run_tag": matrix_run_tag,
            "aggregate": to_json_safe(aggregate),
            "episodes": [
                {
                    "episode": ep["episode"],
                    "steps": ep["steps"],
                    "total_reward": ep["total_reward"],
                    "initial_qor": ep["initial_qor"],
                    "final_qor": ep["final_qor"],
                    "qor_delta": ep["qor_delta"],
                    "final_circuit": ep["final_circuit"],
                    "final_state_file": ep["final_state_file"],
                }
                for ep in all_episodes
            ],
        }
        summary_path.write_text(json.dumps(to_json_safe(summary_payload), indent=2), encoding="utf-8")

        with csv_path.open("w", newline="", encoding="utf-8") as f:
            fieldnames = ["episode", "step", "action_id", "action_name", "reward", "qor_prev", "qor_next", "state_file", "circuit", "lut6_count", "lut6_lev"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)

        print("=== Inference summary ===")
        print(f"mean_total_reward: {aggregate['mean_total_reward']}")
        print(f"mean_final_qor: {aggregate['mean_final_qor']}")
        print(f"best_episode_by_reward: {aggregate['best_episode_by_reward']}")
        print(f"best_episode_by_final_qor: {aggregate['best_episode_by_final_qor']}")
        print(f"Saved trace JSON: {trace_path}")
        print(f"Saved summary JSON: {summary_path}")
        print(f"Saved steps CSV: {csv_path}")
        print("=== Inference run end ===")
    finally:
        if env is not None:
            env.close()
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
        log_file.close()


if __name__ == "__main__":
    main()