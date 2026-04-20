#!/usr/bin/env python3
"""Continue PPO training from a saved model/checkpoint.

This resume entrypoint keeps the same artifact behavior as main training:
- mirrored console + log file output
- eval callback (best model + eval metrics)
- checkpoint callback (--save-freq)
- tensorboard logging in the run folder
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import TextIO

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

from hybridsyn_env import HybridSYNEnv
from train_hybridsyn_ppo import TrainingProgressCallback


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


def infer_run_dir_from_model(model_path: Path) -> Path:
	if model_path.parent.name in {"best", "checkpoints"}:
		return model_path.parent.parent
	return model_path.parent


def choose_resume_log_file(run_dir: Path, explicit_log_file: str) -> Path:
	if explicit_log_file:
		explicit = Path(explicit_log_file)
		if explicit.suffix:
			return run_dir / explicit.name
		return run_dir / f"{explicit.name}.log"

	preferred = run_dir / f"{run_dir.name}.log"
	if preferred.exists():
		return preferred

	existing_logs = sorted(run_dir.glob("*.log"))
	if existing_logs:
		return existing_logs[0]

	return run_dir / "continue_training.log"


def load_run_args(run_dir: Path) -> dict:
	run_args_path = run_dir / "run_args.json"
	if not run_args_path.exists():
		return {}
	try:
		return json.loads(run_args_path.read_text(encoding="utf-8"))
	except (OSError, ValueError, json.JSONDecodeError):
		return {}


def make_env(
	circuit_file: Path,
	mode: str,
	circuit_pool: list[Path],
	matrix_run_tag: str,
	max_steps: int,
	seed: int,
	reward_alpha: float,
	reward_beta: float,
	step_penalty: float,
	reward_scale: float,
	reward_clip: float,
):
	env = HybridSYNEnv(
		circuit_file=str(circuit_file),
		mode=mode,
		circuit_pool_files=[str(p) for p in circuit_pool],
		matrix_run_tag=matrix_run_tag,
		max_steps=max_steps,
		seed=seed,
		reward_alpha=reward_alpha,
		reward_beta=reward_beta,
		step_penalty=step_penalty,
		reward_scale=reward_scale,
		reward_clip=reward_clip,
	)
	return Monitor(env)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Continue PPO training from a saved checkpoint/model")
	parser.add_argument("--model-path", type=str, required=True, help="Path to checkpoint/best/final model zip")
	parser.add_argument("--mode", type=str, choices=["auto", "single", "multiple"], default="auto")
	parser.add_argument("--circuit-file", type=str, default=DEFAULT_CIRCUIT_FILE)
	parser.add_argument("--circuit-pool", nargs="+", default=[])
	parser.add_argument("--additional-actions", type=int, default=10_000, help="Additional timesteps to train")
	parser.add_argument("--sequence-steps", type=int, default=20)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--device", type=str, default="auto", help="auto, cpu, or cuda")
	parser.add_argument("--matrix-run-tag", type=str, default="", help="Override run tag used for matrix/320D outputs")
	parser.add_argument("--run-dir", type=str, default="", help="Override run artifact directory")
	parser.add_argument("--resume-log-file", type=str, default="", help="Log filename to append/create in run-dir")
	parser.add_argument("--output-model-name", type=str, default="ppo_hybridsyn_resumed_final")

	parser.add_argument("--reward-alpha", type=float, default=0.5, help="LUT6 weight in QoR formula (default 0.5)")
	parser.add_argument("--reward-beta", type=float, default=0.5, help="Level weight in QoR formula (default 0.5)")
	parser.add_argument("--step-penalty", type=float, default=0.0, help="Penalty per step (default 0.0)")
	parser.add_argument("--reward-scale", type=float, default=50.0, help="Reward scaling factor for numerical stability (default 50.0)")
	parser.add_argument("--reward-clip", type=float, default=10.0, help="Reward clipping range (default 10.0)")

	parser.add_argument("--eval-freq", type=int, default=5000)
	parser.add_argument("--eval-episodes", type=int, default=5)
	parser.add_argument("--save-freq", type=int, default=10_000)
	parser.add_argument("--progress-bar", action="store_true")
	return parser.parse_args()


def main() -> None:
	args = parse_args()

	model_path = resolve_path(args.model_path)
	if not model_path.exists():
		raise FileNotFoundError(f"Model path not found: {model_path}")

	if args.run_dir:
		run_dir = resolve_path(args.run_dir)
	else:
		run_dir = infer_run_dir_from_model(model_path)
	run_dir.mkdir(parents=True, exist_ok=True)
	run_args = load_run_args(run_dir)

	if args.mode == "auto":
		mode = str(run_args.get("mode", "single"))
	else:
		mode = args.mode

	matrix_run_tag = args.matrix_run_tag or run_dir.name
	checkpoint_dir = run_dir / "checkpoints"
	best_dir = run_dir / "best"
	eval_dir = run_dir / "eval"

	if mode == "single":
		if args.mode == "auto" and args.circuit_file == DEFAULT_CIRCUIT_FILE:
			circuit_file = resolve_path(str(run_args.get("circuit_file", DEFAULT_CIRCUIT_FILE)))
		else:
			circuit_file = resolve_path(args.circuit_file)
		circuit_pool: list[Path] = []
	else:
		pool_values = args.circuit_pool or run_args.get("circuit_pool", []) or []
		if not pool_values:
			raise ValueError("multiple mode requires --circuit-pool or run_args.json with circuit_pool")
		circuit_pool = [resolve_path(str(item)) for item in pool_values]
		circuit_file = circuit_pool[0]

	run_log_path = choose_resume_log_file(run_dir, args.resume_log_file)
	continue_args_path = run_dir / "continue_args.json"

	continue_args_path.write_text(json.dumps(vars(args), indent=2, sort_keys=True), encoding="utf-8")

	orig_stdout = sys.stdout
	orig_stderr = sys.stderr
	log_file = run_log_path.open("a", encoding="utf-8")
	sys.stdout = TeeStream(orig_stdout, log_file)
	sys.stderr = TeeStream(orig_stderr, log_file)

	train_env = None
	eval_env = None
	try:
		print("=== Resume training begin ===")
		print(f"Model loaded from: {model_path}")
		print(f"Run dir: {run_dir}")
		print(f"Run log file: {run_log_path}")
		print(f"Continue args file: {continue_args_path}")
		print(f"Mode: {mode}")
		print(f"Matrix run tag: {matrix_run_tag}")
		print(f"Best model dir: {best_dir}")
		print(f"Eval dir: {eval_dir}")
		print(f"Checkpoint dir: {checkpoint_dir}")

		train_env = make_env(
			circuit_file=circuit_file,
			mode=mode,
			circuit_pool=circuit_pool,
			matrix_run_tag=matrix_run_tag,
			max_steps=args.sequence_steps,
			seed=args.seed,
			reward_alpha=args.reward_alpha,
			reward_beta=args.reward_beta,
			step_penalty=args.step_penalty,
			reward_scale=args.reward_scale,
			reward_clip=args.reward_clip,
		)
		eval_env = make_env(
			circuit_file=circuit_file,
			mode=mode,
			circuit_pool=circuit_pool,
			matrix_run_tag=matrix_run_tag,
			max_steps=args.sequence_steps,
			seed=args.seed + 1,
			reward_alpha=args.reward_alpha,
			reward_beta=args.reward_beta,
			step_penalty=args.step_penalty,
			reward_scale=args.reward_scale,
			reward_clip=args.reward_clip,
		)

		model = PPO.load(str(model_path), env=train_env, device=args.device)
		model.set_logger(configure(str(run_dir), ["stdout", "tensorboard"]))

		eval_callback = EvalCallback(
			eval_env,
			best_model_save_path=str(best_dir),
			log_path=str(eval_dir),
			eval_freq=max(1, args.eval_freq),
			n_eval_episodes=args.eval_episodes,
			deterministic=True,
		)
		checkpoint_callback = CheckpointCallback(
			save_freq=max(1, args.save_freq),
			save_path=str(checkpoint_dir),
			name_prefix="checkpoint",
			save_replay_buffer=False,
			save_vecnormalize=False,
		)
		progress_callback = TrainingProgressCallback(
			total_actions=args.additional_actions,
			ppo_steps=max(1, int(getattr(model, "n_steps", 1))),
		)

		model.learn(
			total_timesteps=args.additional_actions,
			reset_num_timesteps=False,
			callback=CallbackList([eval_callback, checkpoint_callback, progress_callback]),
			progress_bar=args.progress_bar,
		)

		output_model = run_dir / args.output_model_name
		model.save(str(output_model))
		print(f"Saved resumed model to: {output_model}.zip")
		print("=== Resume training end ===")
	finally:
		if train_env is not None:
			train_env.close()
		if eval_env is not None:
			eval_env.close()
		sys.stdout = orig_stdout
		sys.stderr = orig_stderr
		log_file.close()


if __name__ == "__main__":
	main()