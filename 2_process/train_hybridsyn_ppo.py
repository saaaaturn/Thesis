#!/usr/bin/env python3
"""Train PPO on HybridSYN environment using Stable-Baselines3.

This script wires your 320D observation + 12-action environment into PPO with
an architecture aligned to your design discussion:
- shared trunk: 320 -> 256 -> 128
- policy/value heads: 128 -> 64 -> output
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, TextIO

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from hybridsyn_env import HybridSYNEnv


SCRIPT_DIR = Path(__file__).resolve().parent


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


class SharedTrunkExtractor(BaseFeaturesExtractor):
    """Feature extractor implementing shared 320 -> 256 -> 128 trunk."""

    def __init__(self, observation_space: spaces.Box, features_dim: int = 128) -> None:
        super().__init__(observation_space, features_dim)
        in_dim = int(observation_space.shape[0])
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)


class TrainingProgressCallback(BaseCallback):
    """Print step-level and sequence-level progress during PPO training."""

    def __init__(self, total_actions: int, ppo_steps: int) -> None:
        super().__init__()
        self.total_actions = total_actions
        self.ppo_steps = max(1, ppo_steps)
        self.total_iterations = max(1, math.ceil(total_actions / self.ppo_steps))
        self.iteration_width = max(2, len(str(self.total_iterations)))
        self.sequence_width = 3
        self.step_width = 3
        self.iteration_index = 0
        self.sequence_index = 0
        self.current_sequence_actions: list[str] = []
        self.current_sequence_begin_qor: float | None = None
        self.current_sequence_best_qor: float | None = None
        self.current_sequence_last_lut6_count: float | None = None
        self.current_sequence_last_lut6_lev: float | None = None
        self.current_sequence_best_lut6_count: float | None = None
        self.current_sequence_best_lut6_lev: float | None = None
        self.current_sequence_qors: list[float] = []
        self.current_sequence_id: int | None = None
        self.current_iteration_sequences: list[dict[str, object]] = []
        self.current_iteration_qors: list[float] = []
        self.current_iteration_lut6_counts: list[float] = []
        self.current_iteration_lut6_levels: list[float] = []
        self.action_lut6_deltas: dict[str, list[float]] = {}
        self.total_action_exec = 0
        self.total_action_exec_fail = 0
        self.total_reencode_attempt = 0
        self.total_reencode_fail = 0
        self.total_clean_signal_breach = 0

    @staticmethod
    def _format_qor(value: object) -> str:
        if value is None:
            return "n/a"
        try:
            return f"{float(value):.4f}"
        except (TypeError, ValueError):
            return str(value)

    @staticmethod
    def _format_reward(value: object) -> str:
        try:
            return f"{float(value):+.6f}"
        except (TypeError, ValueError):
            return str(value)

    @staticmethod
    def _to_float_or_none(value: object) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _format_state_file(value: object) -> str:
        if value is None:
            return "n/a"
        try:
            return Path(str(value)).name
        except (TypeError, ValueError):
            return str(value)

    def _on_training_start(self) -> None:
        print("=== HybridSYN PPO training begin ===")

    def _on_rollout_start(self) -> None:
        self.iteration_index += 1
        self.current_iteration_sequences = []
        self.current_iteration_qors = []
        self.current_iteration_lut6_counts = []
        self.current_iteration_lut6_levels = []
        print(
            f"=== Iteration {self.iteration_index:0{self.iteration_width}d}/{self.total_iterations:0{self.iteration_width}d} begin ==="
        )

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        if not infos:
            return True

        info = infos[0]
        actions = self.locals.get("actions")
        rewards = self.locals.get("rewards")
        dones = self.locals.get("dones")

        action_array = np.asarray(actions).reshape(-1) if actions is not None else np.asarray([])
        reward_array = np.asarray(rewards).reshape(-1) if rewards is not None else np.asarray([])
        done_array = np.asarray(dones).reshape(-1) if dones is not None else np.asarray([])

        action_value = action_array[0] if action_array.size else None
        reward_value = reward_array[0] if reward_array.size else None
        done_value = bool(done_array[0]) if done_array.size else False

        step_index = int(info.get("step", 0))
        action_name = str(info.get("action_name", action_value))
        current_qor = info.get("qor_next", info.get("score"))
        previous_qor = info.get("qor_prev")
        previous_metrics = info.get("metrics_prev") or {}
        state_file_before = info.get("state_file_before")
        state_file = info.get("state_file")
        metrics_next = info.get("metrics_next") or {}
        and_count = metrics_next.get("and_count", "n/a")
        inv_count = metrics_next.get("inv_count", "n/a")
        depth = metrics_next.get("lev", "n/a")
        lut6_count = metrics_next.get("lut6_count", "n/a")
        lut6_lev = metrics_next.get("lut6_lev", "n/a")
        lut6_count_float = self._to_float_or_none(lut6_count)
        lut6_lev_float = self._to_float_or_none(lut6_lev)
        current_qor_float = self._to_float_or_none(current_qor)
        if current_qor_float is not None:
            self.current_iteration_qors.append(current_qor_float)
            self.logger.record("qor/current", current_qor_float)
        if lut6_count_float is not None:
            self.current_iteration_lut6_counts.append(lut6_count_float)
            self.logger.record("lut6/current", lut6_count_float)
        if lut6_lev_float is not None:
            self.current_iteration_lut6_levels.append(lut6_lev_float)
            self.logger.record("lut6_level/current", lut6_lev_float)

        prev_lut6_float = self._to_float_or_none(previous_metrics.get("lut6_count"))
        if prev_lut6_float is not None and lut6_count_float is not None:
            # Positive delta means action reduced LUT6 count.
            lut6_delta = prev_lut6_float - lut6_count_float
            self.action_lut6_deltas.setdefault(action_name, []).append(lut6_delta)

        action_exec = info.get("action_exec") or {}
        if action_exec.get("executed"):
            self.total_action_exec += 1
            if not action_exec.get("success"):
                self.total_action_exec_fail += 1

        state_reencode = info.get("state_reencode")
        if state_reencode is not None:
            self.total_reencode_attempt += 1
            if not state_reencode.get("success"):
                self.total_reencode_fail += 1

        if info.get("clean_signal_breach"):
            self.total_clean_signal_breach += 1

        if step_index == 1 or self.current_sequence_id is None:
            self.sequence_index += 1
            self.current_sequence_id = self.sequence_index
            self.current_sequence_actions = []
            self.current_sequence_begin_qor = previous_qor
            self.current_sequence_best_qor = current_qor_float
            self.current_sequence_last_lut6_count = lut6_count_float
            self.current_sequence_last_lut6_lev = lut6_lev_float
            self.current_sequence_best_lut6_count = lut6_count_float
            self.current_sequence_best_lut6_lev = lut6_lev_float
            self.current_sequence_qors = []
            if current_qor_float is not None:
                self.current_sequence_qors.append(current_qor_float)
            base_circuit = info.get("base_circuit")
            if base_circuit:
                print(
                    f"--- Sequence {self.current_sequence_id:0{self.sequence_width}d} base circuit: "
                    f"{Path(str(base_circuit)).name} ---"
                )
            print(
                f"--- Sequence {self.current_sequence_id:0{self.sequence_width}d} loaded 320D: "
                f"{self._format_state_file(state_file_before or state_file)} ---"
            )
            print(
                f"--- Sequence {self.current_sequence_id:0{self.sequence_width}d} begin | beginning QoR: {self._format_qor(previous_qor)} ---"
            )
            print(
                f"--- Sequence {self.current_sequence_id:0{self.sequence_width}d} begin | "
                f"initial LUT6: {self._format_qor(previous_metrics.get('lut6_count'))} | "
                f"initial Level: {self._format_qor(previous_metrics.get('lut6_lev'))} ---"
            )
        else:
            self.current_sequence_last_lut6_count = lut6_count_float
            self.current_sequence_last_lut6_lev = lut6_lev_float
            if current_qor_float is not None:
                self.current_sequence_qors.append(current_qor_float)
                if self.current_sequence_best_qor is None or current_qor_float > self.current_sequence_best_qor:
                    self.current_sequence_best_qor = current_qor_float
                    self.current_sequence_best_lut6_count = lut6_count_float
                    self.current_sequence_best_lut6_lev = lut6_lev_float

        self.current_sequence_actions.append(action_name)

        print(
            f"[Iteration {self.iteration_index:0{self.iteration_width}d}/{self.total_iterations:0{self.iteration_width}d}] "
            f"[Sequence {self.current_sequence_id:0{self.sequence_width}d}] "
            f"[Step {step_index:0{self.step_width}d}] "
            f"Action={action_name} "
            f"Current QoR={self._format_qor(current_qor)} "
            f"Reward={self._format_reward(reward_value)} "
            f"Area(AND={and_count}, INV={inv_count}) "
            f"Depth={depth} "
            f"LUT6={lut6_count} LUT6_Level={lut6_lev}"
        )

        if done_value and self.current_sequence_id is not None:
            sequence_avg_qor = None
            if self.current_sequence_qors:
                sequence_avg_qor = sum(self.current_sequence_qors) / len(self.current_sequence_qors)
            begin_qor_float = self._to_float_or_none(self.current_sequence_begin_qor)
            best_qor_float = self._to_float_or_none(self.current_sequence_best_qor)
            sequence_record = {
                "sequence_id": self.current_sequence_id,
                "actions": list(self.current_sequence_actions),
                "begin_qor": self.current_sequence_begin_qor,
                "best_qor": self.current_sequence_best_qor,
                "avg_qor": sequence_avg_qor,
                "last_lut6_count": self.current_sequence_last_lut6_count,
                "last_lut6_lev": self.current_sequence_last_lut6_lev,
                "best_lut6_count": self.current_sequence_best_lut6_count,
                "best_lut6_lev": self.current_sequence_best_lut6_lev,
            }
            self.current_iteration_sequences.append(sequence_record)
            if begin_qor_float is not None:
                self.logger.record("qor/sequence_begin", begin_qor_float)
            if best_qor_float is not None:
                self.logger.record("qor/sequence_best", best_qor_float)
            if sequence_avg_qor is not None:
                self.logger.record("qor/sequence_avg", sequence_avg_qor)
            last_lut6_float = self._to_float_or_none(self.current_sequence_last_lut6_count)
            last_lev_float = self._to_float_or_none(self.current_sequence_last_lut6_lev)
            best_lut6_float = self._to_float_or_none(self.current_sequence_best_lut6_count)
            best_lev_float = self._to_float_or_none(self.current_sequence_best_lut6_lev)
            if last_lut6_float is not None:
                self.logger.record("lut6/sequence_last", last_lut6_float)
            if last_lev_float is not None:
                self.logger.record("lut6_level/sequence_last", last_lev_float)
            if best_lut6_float is not None:
                self.logger.record("lut6/sequence_best", best_lut6_float)
            if best_lev_float is not None:
                self.logger.record("lut6_level/sequence_best", best_lev_float)
            print(
                f"--- Sequence {self.current_sequence_id:0{self.sequence_width}d} end | "
                f"beginning QoR: {self._format_qor(self.current_sequence_begin_qor)} | "
                f"best QoR: {self._format_qor(self.current_sequence_best_qor)} | "
                f"average QoR: {self._format_qor(sequence_avg_qor)} | "
                f"last LUT6: {self._format_qor(self.current_sequence_last_lut6_count)} | "
                f"last LUT6 Level: {self._format_qor(self.current_sequence_last_lut6_lev)} ---"
            )
            self.current_sequence_id = None
            self.current_sequence_actions = []
            self.current_sequence_begin_qor = None
            self.current_sequence_best_qor = None
            self.current_sequence_last_lut6_count = None
            self.current_sequence_last_lut6_lev = None
            self.current_sequence_best_lut6_count = None
            self.current_sequence_best_lut6_lev = None
            self.current_sequence_qors = []

        return True

    def _on_rollout_end(self) -> None:
        finished_sequences = len(self.current_iteration_sequences)
        print(
            f"=== Iteration {self.iteration_index:0{self.iteration_width}d}/{self.total_iterations:0{self.iteration_width}d} end ==="
        )
        print(
            f"Finished {finished_sequences} sequence(s) in iteration {self.iteration_index:0{self.iteration_width}d}"
        )
        if finished_sequences:
            print("Actions taken:")
            for sequence_record in self.current_iteration_sequences:
                action_list = " -> ".join(sequence_record["actions"])
                print(
                    f"  sequence {int(sequence_record['sequence_id']):0{self.sequence_width}d}: {action_list}"
                )
            first_begin_qor = self.current_iteration_sequences[0]["begin_qor"]
            iteration_best_qor = None
            iteration_best_sequence_record: dict[str, object] | None = None
            for sequence_record in self.current_iteration_sequences:
                sequence_best_qor = self._to_float_or_none(sequence_record.get("best_qor"))
                if sequence_best_qor is not None:
                    if iteration_best_qor is None or sequence_best_qor > iteration_best_qor:
                        iteration_best_qor = sequence_best_qor
                        iteration_best_sequence_record = sequence_record
            iteration_avg_qor = None
            if self.current_iteration_qors:
                iteration_avg_qor = sum(self.current_iteration_qors) / len(self.current_iteration_qors)
            iteration_avg_lut6 = None
            if self.current_iteration_lut6_counts:
                iteration_avg_lut6 = sum(self.current_iteration_lut6_counts) / len(self.current_iteration_lut6_counts)
            iteration_avg_lut6_lev = None
            if self.current_iteration_lut6_levels:
                iteration_avg_lut6_lev = sum(self.current_iteration_lut6_levels) / len(self.current_iteration_lut6_levels)
            begin_qor_float = self._to_float_or_none(first_begin_qor)
            best_qor_float = self._to_float_or_none(iteration_best_qor)
            avg_qor_float = self._to_float_or_none(iteration_avg_qor)
            if begin_qor_float is not None:
                self.logger.record("qor/iteration_begin", begin_qor_float)
            if best_qor_float is not None:
                self.logger.record("qor/iteration_best", best_qor_float)
            if avg_qor_float is not None:
                self.logger.record("qor/iteration_avg", avg_qor_float)
            avg_lut6_float = self._to_float_or_none(iteration_avg_lut6)
            avg_lev_float = self._to_float_or_none(iteration_avg_lut6_lev)
            if avg_lut6_float is not None:
                self.logger.record("lut6/iteration_avg", avg_lut6_float)
            if avg_lev_float is not None:
                self.logger.record("lut6_level/iteration_avg", avg_lev_float)
            print(f"Beginning QoR: {self._format_qor(first_begin_qor)}")
            print(f"Best QoR: {self._format_qor(iteration_best_qor)}")
            print(f"Average QoR: {self._format_qor(iteration_avg_qor)}")
            print(f"Average LUT6: {self._format_qor(iteration_avg_lut6)}")
            print(f"Average LUT6 Level: {self._format_qor(iteration_avg_lut6_lev)}")
            if iteration_best_sequence_record is not None:
                print(
                    f"Best-sequence LUT6: {self._format_qor(iteration_best_sequence_record.get('best_lut6_count'))}"
                )
                print(
                    f"Best-sequence LUT6 Level: {self._format_qor(iteration_best_sequence_record.get('best_lut6_lev'))}"
                )

        if self.action_lut6_deltas:
            action_avg_deltas: list[tuple[str, float]] = []
            for action_name, deltas in self.action_lut6_deltas.items():
                if deltas:
                    action_avg_deltas.append((action_name, sum(deltas) / len(deltas)))
            action_avg_deltas.sort(key=lambda item: item[1], reverse=True)
            top_actions = action_avg_deltas[:3]
            bottom_actions = action_avg_deltas[-3:]
            if top_actions:
                print("Top LUT6-reducing actions this run:")
                for action_name, avg_delta in top_actions:
                    print(f"  {action_name}: avg LUT6 delta {avg_delta:+.4f}")
            if bottom_actions:
                print("Most LUT6-harmful actions this run:")
                for action_name, avg_delta in bottom_actions:
                    print(f"  {action_name}: avg LUT6 delta {avg_delta:+.4f}")

        if self.total_action_exec > 0:
            exec_fail_rate = self.total_action_exec_fail / self.total_action_exec
            self.logger.record("pipeline/action_exec_fail_rate", exec_fail_rate)
            print(
                f"Pipeline action execution: {self.total_action_exec_fail}/{self.total_action_exec} failed "
                f"({exec_fail_rate:.2%})"
            )

        if self.total_reencode_attempt > 0:
            reencode_fail_rate = self.total_reencode_fail / self.total_reencode_attempt
            self.logger.record("pipeline/reencode_fail_rate", reencode_fail_rate)
            print(
                f"Pipeline state re-encode: {self.total_reencode_fail}/{self.total_reencode_attempt} failed "
                f"({reencode_fail_rate:.2%})"
            )

        self.logger.record("pipeline/clean_signal_breach_total", self.total_clean_signal_breach)
        print(f"Clean-signal breaches observed: {self.total_clean_signal_breach}")

    def _on_training_end(self) -> None:
        print("=== HybridSYN PPO training end ===")


class LUT6EvalCallback(BaseCallback):
    """Evaluate by final LUT6 and save best checkpoint by lowest mean LUT6."""

    def __init__(
        self,
        eval_env: gym.Env,
        save_dir: Path,
        eval_freq: int = 5000,
        n_eval_episodes: int = 5,
        deterministic: bool = True,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.eval_env = eval_env
        self.save_dir = save_dir
        self.eval_freq = max(1, int(eval_freq))
        self.n_eval_episodes = max(1, int(n_eval_episodes))
        self.deterministic = deterministic
        self.seed = seed
        self.best_mean_lut6 = float("inf")
        self.best_mean_lut6_level = float("inf")
        self.save_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _safe_float(value: object) -> float | None:
        try:
            if value is None:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    def _evaluate_once(self) -> tuple[float | None, float | None, int]:
        lut6_values: list[float] = []
        lut6_level_values: list[float] = []

        for ep in range(self.n_eval_episodes):
            eval_seed = None if self.seed is None else (self.seed + self.n_calls + ep)
            obs, info = self.eval_env.reset(seed=eval_seed)
            done = False
            final_info: dict[str, Any] = {}

            while not done:
                action, _ = self.model.predict(obs, deterministic=self.deterministic)
                obs, _reward, terminated, truncated, step_info = self.eval_env.step(int(action))
                final_info = step_info
                done = bool(terminated or truncated)

            metrics = (final_info.get("metrics_next") if final_info else None) or (info.get("metrics") or {})
            lut6_val = self._safe_float(metrics.get("lut6_count"))
            lut6_level_val = self._safe_float(metrics.get("lut6_lev"))
            if lut6_val is not None:
                lut6_values.append(lut6_val)
            if lut6_level_val is not None:
                lut6_level_values.append(lut6_level_val)

        mean_lut6 = (sum(lut6_values) / len(lut6_values)) if lut6_values else None
        mean_lut6_level = (sum(lut6_level_values) / len(lut6_level_values)) if lut6_level_values else None
        return mean_lut6, mean_lut6_level, len(lut6_values)

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True

        mean_lut6, mean_lut6_level, valid_count = self._evaluate_once()
        if mean_lut6 is not None:
            self.logger.record("eval_lut6/mean", mean_lut6)
        if mean_lut6_level is not None:
            self.logger.record("eval_lut6_level/mean", mean_lut6_level)
        self.logger.record("eval_lut6/valid_episodes", valid_count)

        print(
            f"[LUT6 Eval @ step {self.n_calls}] mean LUT6={mean_lut6} "
            f"mean LUT6_Level={mean_lut6_level} valid={valid_count}/{self.n_eval_episodes}"
        )

        if mean_lut6 is None:
            return True

        if mean_lut6 < self.best_mean_lut6:
            self.best_mean_lut6 = mean_lut6
            if mean_lut6_level is not None:
                self.best_mean_lut6_level = mean_lut6_level
            model_path = self.save_dir / "best_lut6_model"
            self.model.save(str(model_path))
            meta_path = self.save_dir / "best_lut6_meta.json"
            meta_payload = {
                "step": self.n_calls,
                "best_mean_lut6": self.best_mean_lut6,
                "best_mean_lut6_level": self.best_mean_lut6_level,
                "n_eval_episodes": self.n_eval_episodes,
            }
            meta_path.write_text(json.dumps(meta_payload, indent=2), encoding="utf-8")
            print(
                f"[LUT6 Eval] New best checkpoint saved: {model_path}.zip "
                f"(mean LUT6={self.best_mean_lut6:.4f})"
            )

        return True


class MultiModePerCircuitEvalCallback(BaseCallback):
    """Evaluate trained model on each circuit individually for multi-mode clarity."""

    def __init__(
        self, 
        circuit_pool: list[Path],
        mode: str,
        matrix_run_tag: str,
        max_steps: int,
        seed: int,
        reward_alpha: float,
        reward_beta: float,
        step_penalty: float,
        reward_scale: float,
        reward_clip: float,
        allowed_actions: list[str],
        enforce_clean_signal: bool,
        eval_freq: int = 5000,
        n_episodes: int = 1
    ) -> None:
        super().__init__()
        self.circuit_pool = circuit_pool
        self.mode = mode
        self.matrix_run_tag = matrix_run_tag
        self.max_steps = max_steps
        self.seed = seed
        self.reward_alpha = reward_alpha
        self.reward_beta = reward_beta
        self.step_penalty = step_penalty
        self.reward_scale = reward_scale
        self.reward_clip = reward_clip
        self.allowed_actions = allowed_actions
        self.enforce_clean_signal = enforce_clean_signal
        self.eval_freq = eval_freq
        self.n_episodes = n_episodes
        self.n_calls = 0

    def _on_step(self) -> bool:
        if self.eval_freq <= 0 or self.n_calls % self.eval_freq != 0:
            self.n_calls += 1
            return True

        self.logger.record("eval/checkpoint_step", self.n_calls)

        circuit_qors: dict[str, list[float]] = {}
        for circuit_path in self.circuit_pool:
            circuit_name = circuit_path.stem
            circuit_qors[circuit_name] = []

            for episode in range(self.n_episodes):
                eval_env = HybridSYNEnv(
                    circuit_file=str(circuit_path),
                    mode="single",
                    circuit_pool_files=[],
                    matrix_run_tag=self.matrix_run_tag,
                    max_steps=self.max_steps,
                    seed=self.seed + episode if self.seed is not None else None,
                    reward_alpha=self.reward_alpha,
                    reward_beta=self.reward_beta,
                    step_penalty=self.step_penalty,
                    reward_scale=self.reward_scale,
                    reward_clip=self.reward_clip,
                    allowed_actions=self.allowed_actions,
                    enforce_clean_signal=self.enforce_clean_signal,
                )
                
                obs, info = eval_env.reset()
                done = False
                episode_qor = None

                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = eval_env.step(action)
                    episode_qor = info.get("qor_next")
                    done = terminated or truncated

                if episode_qor is not None:
                    circuit_qors[circuit_name].append(episode_qor)

                eval_env.close()

        # Log per-circuit final QoR and compute macro average
        all_qors: list[float] = []
        for circuit_name, qor_list in circuit_qors.items():
            if qor_list:
                avg_qor = sum(qor_list) / len(qor_list)
                self.logger.record(f"eval/circuit_{circuit_name}_qor", avg_qor)
                all_qors.extend(qor_list)

        if all_qors:
            macro_avg = sum(all_qors) / len(all_qors)
            self.logger.record("eval/macro_qor_avg", macro_avg)
            print(
                f"[Multi-mode Eval @ step {self.n_calls}] Per-circuit QoR: {circuit_qors}, "
                f"Macro-average: {macro_avg:.4f}"
            )

        self.n_calls += 1
        return True


def print_run_configuration(args: argparse.Namespace, log_dir: Path, model_dir: Path) -> None:
    print("=== Run options ===")
    ordered_keys = [
        "mode",
        "circuit_file",
        "circuit_pool",
        "sequence_steps",
        "total_actions",
        "ppo_steps",
        "batch_size",
        "learning_rate",
        "gamma",
        "gae_lambda",
        "clip_range",
        "ent_coef",
        "vf_coef",
        "seed",
        "device",
        "reward_alpha",
        "reward_beta",
        "step_penalty",
        "reward_scale",
        "reward_clip",
        "allowed_actions",
        "enforce_clean_signal",
        "best_by",
        "log_file_name",
        "matrix_run_tag",
        "eval_freq",
        "eval_episodes",
        "save_freq",
        "progress_bar",
    ]
    for key in ordered_keys:
        value = getattr(args, key)
        print(f"{key}: {value}")
    print(f"resolved_log_dir: {log_dir}")
    print(f"resolved_model_dir: {model_dir}")
    print("===================")


def make_env(
    circuit_file: Path,
    mode: str,
    circuit_pool: list[Path],
    matrix_run_tag: str,
    max_steps: int,
    seed: int | None,
    reward_alpha: float,
    reward_beta: float,
    step_penalty: float,
    reward_scale: float,
    reward_clip: float,
    allowed_actions: list[str],
    enforce_clean_signal: bool,
) -> gym.Env:
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
        allowed_actions=allowed_actions,
        enforce_clean_signal=enforce_clean_signal,
    )
    return Monitor(env)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO for HybridSYN")
    parser.add_argument("--mode", type=str, choices=["single", "multiple"], default="single")
    parser.add_argument("--circuit-file", type=str, default="../1_inputs/EPFL_benchmarks/arithmetic/adder.aig")
    parser.add_argument(
        "--circuit-pool",
        nargs="+",
        default=[],
        help="Circuit pool for multiple mode; each new sequence uses the next circuit in order.",
    )
    parser.add_argument(
        "--sequence-steps",
        dest="sequence_steps",
        type=int,
        default=20,
        help="Maximum actions per sequence (episode).",
    )
    parser.add_argument(
        "--total-actions",
        type=int,
        default=100_000,
        help="Total number of environment actions to train on.",
    )
    parser.add_argument(
        "--ppo-steps",
        dest="ppo_steps",
        type=int,
        default=1024,
        help="Actions collected before each PPO update.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.003)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, or cuda")
    parser.add_argument("--reward-alpha", type=float, default=1.0, help="LUT6 weight in QoR formula (default 1.0)")
    parser.add_argument("--reward-beta", type=float, default=0.0, help="Level weight in QoR formula (default 0.0)")
    parser.add_argument("--step-penalty", type=float, default=0.0, help="Penalty per step (default 0.0)")
    parser.add_argument("--reward-scale", type=float, default=10.0, help="Reward scaling factor for numerical stability (default 10.0)")
    parser.add_argument("--reward-clip", type=float, default=10.0, help="Reward clipping range (default 10.0)")
    parser.add_argument(
        "--allowed-actions",
        nargs="+",
        default=[],
        help="Optional subset of action names to enable; useful for pruning harmful actions.",
    )
    parser.add_argument(
        "--enforce-clean-signal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Terminate sequence and penalize when action execution/re-encode pipeline fails.",
    )
    parser.add_argument(
        "--best-by",
        choices=["lut6", "reward", "both"],
        default="lut6",
        help="Criterion for saving best checkpoint(s).",
    )
    parser.add_argument("--log-file-name", type=str, default="train.log")
    parser.add_argument("--eval-freq", type=int, default=5000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--save-freq", type=int, default=10_000)
    parser.add_argument(
        "--progress-bar",
        action="store_true",
        help="Enable SB3 progress bar (requires tqdm and rich).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    circuit_file = resolve_path(args.circuit_file)
    circuit_pool = [resolve_path(item) for item in (args.circuit_pool or [])]
    if args.mode == "multiple" and not circuit_pool:
        raise ValueError("multiple mode requires --circuit-pool with at least one circuit")

    artifacts_root = (SCRIPT_DIR / "../3_outputs").resolve()
    circuit_name = circuit_file.stem
    run_name = Path(args.log_file_name).stem
    args.matrix_run_tag = run_name
    if args.mode == "multiple":
        run_dir = artifacts_root / "MULTIPLE_MODE" / run_name
    else:
        run_dir = artifacts_root / circuit_name / run_name
    log_dir = run_dir
    model_dir = run_dir
    checkpoint_dir = run_dir / "checkpoints"

    run_dir.mkdir(parents=True, exist_ok=True)

    run_args_path = run_dir / "run_args.json"
    run_args_path.write_text(json.dumps(vars(args), indent=2, sort_keys=True), encoding="utf-8")

    log_file_name = args.log_file_name if Path(args.log_file_name).suffix else f"{args.log_file_name}.log"
    run_log_path = run_dir / log_file_name
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    log_file = run_log_path.open("w", encoding="utf-8")
    sys.stdout = TeeStream(orig_stdout, log_file)
    sys.stderr = TeeStream(orig_stderr, log_file)

    print(f"Run log file: {run_log_path}")
    print(f"Run args file: {run_args_path}")
    print(f"Run artifacts dir: {run_dir}")
    print(f"Run checkpoints dir: {checkpoint_dir}")
    print_run_configuration(args, log_dir, model_dir)
    print("=== Training setup complete ===")
    print("=== Building environments and PPO model ===")

    train_env = None
    eval_env = None
    lut6_eval_env = None
    try:
        train_env = make_env(
            circuit_file,
            args.mode,
            circuit_pool,
            run_name,
            args.sequence_steps,
            args.seed,
            args.reward_alpha,
            args.reward_beta,
            args.step_penalty,
            args.reward_scale,
            args.reward_clip,
            args.allowed_actions,
            args.enforce_clean_signal,
        )
        eval_env = make_env(
            circuit_file,
            args.mode,
            circuit_pool,
            run_name,
            args.sequence_steps,
            args.seed + 1,
            args.reward_alpha,
            args.reward_beta,
            args.step_penalty,
            args.reward_scale,
            args.reward_clip,
            args.allowed_actions,
            args.enforce_clean_signal,
        )
        lut6_eval_env = make_env(
            circuit_file,
            args.mode,
            circuit_pool,
            run_name,
            args.sequence_steps,
            args.seed + 2,
            args.reward_alpha,
            args.reward_beta,
            args.step_penalty,
            args.reward_scale,
            args.reward_clip,
            args.allowed_actions,
            args.enforce_clean_signal,
        )

        policy_kwargs = dict(
            features_extractor_class=SharedTrunkExtractor,
            features_extractor_kwargs=dict(features_dim=128),
            net_arch=dict(pi=[64], vf=[64]),
            activation_fn=nn.ReLU,
        )

        model = PPO(
            policy="MlpPolicy",
            env=train_env,
            learning_rate=args.learning_rate,
            n_steps=args.ppo_steps,
            batch_size=args.batch_size,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=None,
            seed=args.seed,
            device=args.device,
        )
        model.set_logger(configure(str(run_dir), ["stdout", "tensorboard"]))

        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(model_dir / "best"),
            log_path=str(log_dir / "eval"),
            eval_freq=args.eval_freq,
            n_eval_episodes=args.eval_episodes,
            deterministic=True,
        )
        lut6_eval_callback = LUT6EvalCallback(
            eval_env=lut6_eval_env,
            save_dir=model_dir / "best_lut6",
            eval_freq=args.eval_freq,
            n_eval_episodes=args.eval_episodes,
            deterministic=True,
            seed=args.seed + 2,
        )

        checkpoint_callback = CheckpointCallback(
            save_freq=args.save_freq,
            save_path=str(checkpoint_dir),
            name_prefix="checkpoint",
            save_replay_buffer=False,
            save_vecnormalize=False,
        )

        training_progress_callback = TrainingProgressCallback(
            total_actions=args.total_actions,
            ppo_steps=args.ppo_steps,
        )

        # Multi-mode per-circuit evaluation callback
        callbacks_list: list[BaseCallback] = [checkpoint_callback, training_progress_callback, lut6_eval_callback]
        if args.best_by in {"reward", "both"}:
            callbacks_list.insert(0, eval_callback)
        if args.mode == "multiple" and circuit_pool:
            multimode_eval_callback = MultiModePerCircuitEvalCallback(
                circuit_pool=circuit_pool,
                mode=args.mode,
                matrix_run_tag=run_name,
                max_steps=args.sequence_steps,
                seed=args.seed + 100 if args.seed is not None else None,
                reward_alpha=args.reward_alpha,
                reward_beta=args.reward_beta,
                step_penalty=args.step_penalty,
                reward_scale=args.reward_scale,
                reward_clip=args.reward_clip,
                allowed_actions=args.allowed_actions,
                enforce_clean_signal=args.enforce_clean_signal,
                eval_freq=args.eval_freq,
                n_episodes=1,
            )
            callbacks_list.append(multimode_eval_callback)

        model.learn(
            total_timesteps=args.total_actions,
            callback=CallbackList(callbacks_list),
            progress_bar=args.progress_bar,
        )

        final_model_path = model_dir / "ppo_hybridsyn_final"
        model.save(str(final_model_path))
        print(f"Saved final model to: {final_model_path}.zip")
        print("=== Training end ===")
    finally:
        if train_env is not None:
            train_env.close()
        if eval_env is not None:
            eval_env.close()
        if lut6_eval_env is not None:
            lut6_eval_env.close()
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
        log_file.close()


if __name__ == "__main__":
    main()
