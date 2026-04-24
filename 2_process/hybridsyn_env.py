#////////////////////////////////////////////////////////////////////////////////
#//Gymnasium environment prototype for PPO integration with HybridSYN project.
#//Observation: 320D float vector (state embedding)
#//Action: 12 discrete synthesis actions
#//ABC actions execute through ABC; MIG actions execute through the native MIG path.
#////////////////////////////////////////////////////////////////////////////////

#!/usr/bin/env python3
"""HybridSYN Gymnasium environment prototype for PPO.

This environment exposes the expected RL interface for your project:
- Observation: 320D float vector (state embedding)
- Action: 12 discrete synthesis actions

ABC actions execute through ABC and MIG actions execute through a native MIG executor.
The downstream state pipeline still re-encodes the resulting circuit into the 320D state vector.
"""

#/////////////////////////////////////////////////////////////////////////////////
#/////////IMPORTS AND DEFINITIONS/////////////////////////////////////////////////
#/////////////////////////////////////////////////////////////////////////////////

from __future__ import annotations

from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

#////////////////////////////////////////////////////////////////////////////////
#/////////HYBRIDSYN ENVIRONMENT DEFINITION///////////////////////////////////////
#////////////////////////////////////////////////////////////////////////////////

#Actions from ABC tool and Mockturtle.
ACTION_NAMES = [
    "abc_balance",
    "abc_rewrite",
    "abc_refactor",
    "abc_resub",
    "abc_drewrite",
    "abc_drefactor",
    "mig_balance",
    "mig_rewrite",
    "mig_refactor",
    "mig_resub",
    "mig_rewrite_z",
    "mig_cleanup",
]

ABC_ACTION_COMMANDS = {
    "abc_balance": "balance",
    "abc_rewrite": "rewrite",
    "abc_refactor": "refactor",
    "abc_resub": "resub",
    "abc_drewrite": "rewrite -z",
    "abc_drefactor": "refactor -z",
}

MIG_ACTION_NAMES = {
    "mig_balance",
    "mig_rewrite",
    "mig_refactor",
    "mig_resub",
    "mig_rewrite_z",
    "mig_cleanup",
}

#Environment simulates state transitions 
#and rewards based on moving a 320D state vector closer to a target,
#with fixed per-action directions and noise for realism. This is a prototype for PPO integration; replace the logic with real circuit synthesis dynamics when ready.
class HybridSYNEnv(gym.Env):
    """320D state / 12-action environment for PPO integration."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        base_state_file: str = "state_320d.txt", #kept for backward compatibility fallback
        circuit_file: str = "../1_inputs/EPFL_benchmarks/arithmetic/adder.aig", #the original circuit file
        mode: str = "single",
        circuit_pool_files: list[str] | None = None,
        abc_binary: str = "abc/abc", #path to the ABC binary
        mig_executor_binary: str = "build/interface", #path to the MIG executor binary
        extract_features_binary: str = "build/extract_features", #path to the feature extraction binary
        gcn_script: str = "gcn_pipeline.py", #path the the gcn function
        matrix_run_tag: str = "default", #subfolder tag used for matrix outputs
        
        abc_timeout_s: int = 30, #wait response from ABC for 30 seconds before timing out
        pipeline_timeout_s: int = 120, #wait response from the whole pipeline (ABC + feature extraction + GCN) for 120 seconds before timing out
        
        enable_real_action_execution: bool = True, #false when we skip execution; true when we run with real ABC/MIG execution
        enable_real_state_reencode: bool = True, #false when we not recreate the 320d state each step to test PPO integration
        enable_real_qor_reward: bool = True, #true if we want to calculate the real reward
        
        reward_alpha: float = 1.0, #alpha value to calculate the reward (QoR)
        reward_beta: float = 0.1, #beta value to calculate the reward (QoR)
        step_penalty: float = 0.001, #penalty for each step
        reward_scale: float = 50.0, #scaling factor for the reward to keep PPO updates numerically stable
        reward_clip: float = 10.0,
        allowed_actions: list[str] | None = None,
        enforce_clean_signal: bool = True,
       
        max_steps: int = 20,
        step_size: float = 0.03,
        transition_noise_std: float = 0.002,
        reset_noise_std: float = 0.01,
        seed: int | None = None,
    ) -> None:
        super().__init__()

        self.state_dim = 320
        all_action_names = list(ACTION_NAMES)
        if allowed_actions:
            unknown_actions = sorted(set(allowed_actions) - set(all_action_names))
            if unknown_actions:
                raise ValueError(f"Unsupported action names in --allowed-actions: {unknown_actions}")
            self._enabled_action_names = [name for name in all_action_names if name in set(allowed_actions)]
            if not self._enabled_action_names:
                raise ValueError("allowed_actions resulted in an empty action set")
        else:
            self._enabled_action_names = all_action_names
        self.num_actions = len(self._enabled_action_names)
        self.max_steps = max_steps
        self.step_size = step_size
        self.transition_noise_std = transition_noise_std
        self.reset_noise_std = reset_noise_std
        self.enable_real_action_execution = enable_real_action_execution
        self.enable_real_state_reencode = enable_real_state_reencode
        self.enable_real_qor_reward = enable_real_qor_reward
        self.abc_timeout_s = abc_timeout_s
        self.pipeline_timeout_s = pipeline_timeout_s
        self.reward_alpha = reward_alpha
        self.reward_beta = reward_beta
        self.step_penalty = step_penalty
        self.reward_scale = reward_scale
        self.reward_clip = reward_clip
        self.enforce_clean_signal = enforce_clean_signal
        self.matrix_run_tag = matrix_run_tag

        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.state_dim,),
            dtype=np.float32,
        )

        self._process_dir = Path(__file__).resolve().parent
        self._rng = np.random.default_rng(seed)

        abc_path = Path(abc_binary)
        if not abc_path.is_absolute():
            abc_path = self._process_dir / abc_path
        self._abc_binary = abc_path.resolve()

        mig_executor_path = Path(mig_executor_binary)
        if not mig_executor_path.is_absolute():
            mig_executor_path = self._process_dir / mig_executor_path
        self._mig_executor_binary = mig_executor_path.resolve()

        extractor_path = Path(extract_features_binary)
        if not extractor_path.is_absolute():
            extractor_path = self._process_dir / extractor_path
        self._extract_features_binary = extractor_path.resolve()

        gcn_path = Path(gcn_script)
        if not gcn_path.is_absolute():
            gcn_path = self._process_dir / gcn_path
        self._gcn_script = gcn_path.resolve()

        circuit_path = Path(circuit_file)
        if not circuit_path.is_absolute():
            circuit_path = self._process_dir / circuit_path
        self._initial_circuit = circuit_path.resolve()
        self.mode = mode.strip().lower()
        if self.mode not in {"single", "multiple"}:
            raise ValueError(f"Unsupported mode: {mode}. Expected 'single' or 'multiple'.")

        resolved_pool: list[Path] = []
        if circuit_pool_files:
            for item in circuit_pool_files:
                pool_path = Path(item)
                if not pool_path.is_absolute():
                    pool_path = self._process_dir / pool_path
                resolved_pool.append(pool_path.resolve())
        if self.mode == "multiple":
            self._circuit_pool = resolved_pool if resolved_pool else [self._initial_circuit]
        else:
            self._circuit_pool = [self._initial_circuit]

        self._circuit_cycle_index = -1
        self._active_base_circuit = self._initial_circuit
        self._current_circuit = self._active_base_circuit
        self._matrix_circuit_stem = self._active_base_circuit.stem
        self._runtime_dir = self._process_dir / "runtime_circuits" / self.matrix_run_tag / self._matrix_circuit_stem
        self._runtime_dir.mkdir(parents=True, exist_ok=True)
        self._gcn_results_dir = self._process_dir / "GCN_results"
        self._gcn_results_dir.mkdir(parents=True, exist_ok=True)
        base_state_path = Path(base_state_file)
        if not base_state_path.is_absolute():
            base_state_path = self._process_dir / base_state_path
        self._base_state = self._load_state_file(base_state_path.resolve())
        self._generated_state_dir = self._gcn_results_dir / self._matrix_circuit_stem / self.matrix_run_tag
        self._generated_state_dir.mkdir(parents=True, exist_ok=True)
        self._current_state_file = self._generated_state_dir / f"{self._matrix_circuit_stem}_step_0000_320d.txt"

        # Fixed per-action direction vectors for stable, learnable dynamics in the prototype.
        raw_effects = self._rng.normal(loc=0.0, scale=1.0, size=(self.num_actions, self.state_dim)).astype(np.float32)
        norms = np.linalg.norm(raw_effects, axis=1, keepdims=True) + 1e-8
        self._action_effects = raw_effects / norms

        # Synthetic objective target; reward is based on moving state closer to this target.
        self._target = np.zeros((self.state_dim,), dtype=np.float32)

        self._state = self._base_state.copy()
        self._step_count = 0
        self._last_metrics: Dict[str, float] = {}
        
        # Baseline metrics for relative QoR calculation (reset per sequence)
        self.lut_init: float | None = None
        self.lev_init: float | None = None

    @staticmethod
    def _load_state_file(path: Path) -> np.ndarray:                         #check for the state file and load the 320d state vector
        if not path.exists():                                               #if the file not exist --> error
            raise FileNotFoundError(f"Missing base state file: {path}")

        raw = path.read_text(encoding="utf-8").strip().split()
        if not raw:                                                         #if the file is empty --> error
            raise ValueError(f"Empty state file: {path}")

        vec = np.asarray([float(x) for x in raw], dtype=np.float32)
        if vec.shape[0] != 320:                                             #if the dimension is not 320 --> error
            raise ValueError(f"Expected 320 values in {path}, found {vec.shape[0]}")
        return vec

    def _score_state(self, state: np.ndarray) -> float:                     #calculate the score of the current state based on its distance to the target state
        """Higher is better: negative L2 distance to target state."""
        return -float(np.linalg.norm(state - self._target))

    def _read_inv_count_from_statistics(self, circuit_file: Path) -> float | None:
        """Read inverter count proxy from extracted AIG statistics if available.

        The extractor writes `num_inv_edges` for complemented AIG fanins.
        This keeps INV accounting non-zero for strashed AIGs where ABC print_stats
        may not expose explicit inverter nodes.
        """
        stats_path = (
            self._process_dir
            / "matrix"
            / "forAIG"
            / self._matrix_circuit_stem
            / self.matrix_run_tag
            / f"{circuit_file.stem}_statistics.txt"
        )
        if not stats_path.exists():
            return None

        try:
            for line in stats_path.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if len(parts) != 2:
                    continue
                if parts[0] == "num_inv_edges":
                    return float(parts[1])
        except (OSError, ValueError):
            return None
        return None

    def _get_abc_metrics(self, circuit_file: Path) -> Dict[str, Any]:       #run ABC to get the metrics of the current circuit
        if not self._abc_binary.exists():                                   #if the ABC binary is not found --> error
            return {"success": False, "reason": f"abc_binary_not_found:{self._abc_binary}"}
        if not circuit_file.exists():                                       #if the circuit file is not found --> error
            return {"success": False, "reason": f"circuit_not_found:{circuit_file}"}

        script = f"read_aiger {circuit_file}; strash; print_stats"          #script ABC command to get AND/INV counts and level for reward calculation
        try:
            proc = subprocess.run(
                [str(self._abc_binary), "-c", script],
                cwd=str(self._process_dir),
                capture_output=True,
                text=True,
                timeout=self.abc_timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return {"success": False, "reason": "abc_metrics_timeout"}      #if ABC takes too long to respond --> error

        text = (proc.stdout or "") + "\n" + (proc.stderr or "")
        m_and = re.search(r"\band\s*=\s*(\d+)", text)
        m_inv = re.search(r"\binv\s*=\s*(\d+)", text)
        if m_inv is None:
            m_inv = re.search(r"\bnot\s*=\s*(\d+)", text)
        m_lev = re.search(r"\blev\s*=\s*(\d+)", text)
        if proc.returncode != 0 or m_and is None or m_lev is None:
            return {
                "success": False,
                "reason": f"abc_metrics_parse_failed_rc:{proc.returncode}",
                "stdout_tail": (proc.stdout or "")[-500:],
                "stderr_tail": (proc.stderr or "")[-500:],
            }

        and_count = float(m_and.group(1))
        inv_count_from_stats = self._read_inv_count_from_statistics(circuit_file)
        if inv_count_from_stats is not None:
            inv_count = inv_count_from_stats
        else:
            inv_count = float(m_inv.group(1)) if m_inv is not None else 0.0
        lev = float(m_lev.group(1))

        lut6_metrics = self._get_lut6_metrics(circuit_file)
        lut6_count = lut6_metrics.get("lut6_count")
        lut6_lev = lut6_metrics.get("lut6_lev")

        return {
            "success": True,
            "reason": "ok",
            "and_count": and_count,
            "inv_count": inv_count,
            "lev": lev,
            "lut6_count": lut6_count,
            "lut6_lev": lut6_lev,
        }

    def _get_lut6_metrics(self, circuit_file: Path) -> Dict[str, Any]:
        """Compute LUT-6 count and mapped level using ABC technology mapping."""
        script = f"read_aiger {circuit_file}; strash; if -K 6; print_stats"
        try:
            proc = subprocess.run(
                [str(self._abc_binary), "-c", script],
                cwd=str(self._process_dir),
                capture_output=True,
                text=True,
                timeout=self.abc_timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return {"success": False, "reason": "abc_lut6_timeout", "lut6_count": None, "lut6_lev": None}

        text = (proc.stdout or "") + "\n" + (proc.stderr or "")
        m_nd = re.search(r"\bnd\s*=\s*(\d+)", text)
        m_lev = re.search(r"\blev\s*=\s*(\d+)", text)

        if proc.returncode != 0:
            return {
                "success": False,
                "reason": f"abc_lut6_failed_rc:{proc.returncode}",
                "lut6_count": None,
                "lut6_lev": None,
                "stdout_tail": (proc.stdout or "")[-500:],
                "stderr_tail": (proc.stderr or "")[-500:],
            }

        lut6_count = float(m_nd.group(1)) if m_nd is not None else None
        lut6_lev = float(m_lev.group(1)) if m_lev is not None else None
        return {
            "success": True,
            "reason": "ok",
            "lut6_count": lut6_count,
            "lut6_lev": lut6_lev,
        }

    def _state_file_for_step(self, step_index: int) -> Path:
        return self._generated_state_dir / f"{self._matrix_circuit_stem}_step_{step_index:04d}_320d.txt"

    def _select_base_circuit_for_reset(self) -> Path:
        if self.mode == "multiple":
            self._circuit_cycle_index = (self._circuit_cycle_index + 1) % len(self._circuit_pool)
            return self._circuit_pool[self._circuit_cycle_index]
        return self._initial_circuit

    def _refresh_active_paths(self) -> None:
        self._matrix_circuit_stem = self._active_base_circuit.stem
        self._runtime_dir = self._process_dir / "runtime_circuits" / self.matrix_run_tag / self._matrix_circuit_stem
        self._runtime_dir.mkdir(parents=True, exist_ok=True)
        self._generated_state_dir = self._gcn_results_dir / self._matrix_circuit_stem / self.matrix_run_tag
        self._generated_state_dir.mkdir(parents=True, exist_ok=True)
        self._current_state_file = self._state_file_for_step(0)

    def _reencode_state_from_circuit(self, circuit_file: Path, step_index: int) -> Dict[str, Any]:
        if not self._extract_features_binary.exists():
            return {"success": False, "reason": f"extract_features_binary_not_found:{self._extract_features_binary}"}
        if not self._gcn_script.exists():
            return {"success": False, "reason": f"gcn_script_not_found:{self._gcn_script}"}

        try:
            extract_proc = subprocess.run(
                [
                    str(self._extract_features_binary),
                    str(circuit_file),
                    self.matrix_run_tag,
                    self._matrix_circuit_stem,
                ],
                cwd=str(self._extract_features_binary.parent),
                capture_output=True,
                text=True,
                timeout=self.pipeline_timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return {"success": False, "reason": "extract_features_timeout"}

        if extract_proc.returncode != 0:
            return {
                "success": False,
                "reason": f"extract_features_failed_rc:{extract_proc.returncode}",
                "stdout_tail": (extract_proc.stdout or "")[-500:],
                "stderr_tail": (extract_proc.stderr or "")[-500:],
            }

        circuit_stem = self._matrix_circuit_stem
        state_file = self._state_file_for_step(step_index)
        try:
            gcn_proc = subprocess.run(
                [
                    sys.executable,
                    str(self._gcn_script),
                    "--circuit",
                    circuit_stem,
                    "--run-tag",
                    self.matrix_run_tag,
                    "--matrix-circuit",
                    self._matrix_circuit_stem,
                    "--matrix-dir",
                    "matrix",
                    "--gcn-results-dir",
                    str(self._gcn_results_dir),
                    "--save-state",
                    str(state_file),
                ],
                cwd=str(self._process_dir),
                capture_output=True,
                text=True,
                timeout=self.pipeline_timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return {"success": False, "reason": "gcn_pipeline_timeout"}

        if gcn_proc.returncode != 0:
            return {
                "success": False,
                "reason": f"gcn_pipeline_failed_rc:{gcn_proc.returncode}",
                "stdout_tail": (gcn_proc.stdout or "")[-500:],
                "stderr_tail": (gcn_proc.stderr or "")[-500:],
            }

        try:
            vec = self._load_state_file(state_file)
        except (FileNotFoundError, ValueError) as exc:
            return {"success": False, "reason": f"state_load_failed:{exc}"}

        return {
            "success": True,
            "reason": "ok",
            "state": vec,
            "state_file": str(state_file),
        }

    def _resolve_state_file_for_reset(self) -> Path:
        if self._current_state_file.exists():
            return self._current_state_file
        return self._process_dir / "state_320d.txt"

    def _run_abc_action(self, action_name: str) -> Dict[str, Any]:

        if not self.enable_real_action_execution:
            return {
                "executed": False,
                "success": False,
                "reason": "real_action_execution_disabled",
                "action_name": action_name,
            }

        if not self._abc_binary.exists():
            return {
                "executed": False,
                "success": False,
                "reason": f"abc_binary_not_found:{self._abc_binary}",
                "action_name": action_name,
            }

        if not self._current_circuit.exists():
            return {
                "executed": False,
                "success": False,
                "reason": f"circuit_not_found:{self._current_circuit}",
                "action_name": action_name,
            }

        abc_op = ABC_ACTION_COMMANDS[action_name]
        out_file = self._runtime_dir / f"step_{self._step_count + 1:04d}_{action_name}.aig"
        script = (
            f"read_aiger {self._current_circuit}; "
            f"strash; "
            f"{abc_op}; "
            f"write_aiger {out_file}"
        )

        try:
            proc = subprocess.run(
                [str(self._abc_binary), "-c", script],
                cwd=str(self._process_dir),
                capture_output=True,
                text=True,
                timeout=self.abc_timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return {
                "executed": True,
                "success": False,
                "reason": "timeout",
                "action_name": action_name,
                "abc_op": abc_op,
                "output_circuit": str(out_file),
            }

        success = proc.returncode == 0 and out_file.exists()
        if success:
            self._current_circuit = out_file

        return {
            "executed": True,
            "success": success,
            "reason": "ok" if success else f"abc_failed_rc:{proc.returncode}",
            "action_name": action_name,
            "abc_op": abc_op,
            "output_circuit": str(out_file),
            "stdout_tail": proc.stdout[-500:] if proc.stdout else "",
            "stderr_tail": proc.stderr[-500:] if proc.stderr else "",
        }

    def _run_mig_action(self, action_name: str) -> Dict[str, Any]:

        if not self.enable_real_action_execution:
            return {
                "executed": False,
                "success": False,
                "reason": "real_action_execution_disabled",
                "action_name": action_name,
            }

        if not self._mig_executor_binary.exists():
            return {
                "executed": False,
                "success": False,
                "reason": f"mig_executor_binary_not_found:{self._mig_executor_binary}",
                "action_name": action_name,
            }

        if not self._current_circuit.exists():
            return {
                "executed": False,
                "success": False,
                "reason": f"circuit_not_found:{self._current_circuit}",
                "action_name": action_name,
            }

        out_file = self._runtime_dir / f"step_{self._step_count + 1:04d}_{action_name}.aig"
        try:
            proc = subprocess.run(
                [
                    str(self._mig_executor_binary),
                    "--action",
                    action_name,
                    "--input",
                    str(self._current_circuit),
                    "--output",
                    str(out_file),
                ],
                cwd=str(self._process_dir),
                capture_output=True,
                text=True,
                timeout=self.pipeline_timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return {
                "executed": True,
                "success": False,
                "reason": "timeout",
                "action_name": action_name,
                "output_circuit": str(out_file),
            }

        success = proc.returncode == 0 and out_file.exists()
        if success:
            self._current_circuit = out_file

        return {
            "executed": True,
            "success": success,
            "reason": "ok" if success else f"mig_failed_rc:{proc.returncode}",
            "action_name": action_name,
            "output_circuit": str(out_file),
            "stdout_tail": proc.stdout[-500:] if proc.stdout else "",
            "stderr_tail": proc.stderr[-500:] if proc.stderr else "",
        }

    def _apply_action(self, state: np.ndarray, action: int) -> np.ndarray:
        """Surrogate transition: directional move + small noise.

        Replace this with your real action execution pipeline:
        1) run synthesis action on circuit
        2) re-extract matrices
        3) run GCN encoder to get new 320D state
        """
        direction = self._action_effects[action]
        noise = self._rng.normal(0.0, self.transition_noise_std, size=state.shape).astype(np.float32)
        next_state = state + self.step_size * direction + noise
        return next_state.astype(np.float32)

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._active_base_circuit = self._select_base_circuit_for_reset()
        self._refresh_active_paths()
        self._current_circuit = self._active_base_circuit
        if self.enable_real_state_reencode:
            state_info = self._reencode_state_from_circuit(self._current_circuit, step_index=0)
            if state_info["success"]:
                self._state = state_info["state"]
                self._current_state_file = Path(state_info["state_file"])
            else:
                fallback_path = self._resolve_state_file_for_reset()
                fallback_state = self._load_state_file(fallback_path)
                self._current_state_file = fallback_path
                noise = self._rng.normal(0.0, self.reset_noise_std, size=fallback_state.shape).astype(np.float32)
                self._state = (fallback_state + noise).astype(np.float32)
        else:
            fallback_path = self._resolve_state_file_for_reset()
            fallback_state = self._load_state_file(fallback_path)
            self._current_state_file = fallback_path
            noise = self._rng.normal(0.0, self.reset_noise_std, size=fallback_state.shape).astype(np.float32)
            self._state = (fallback_state + noise).astype(np.float32)

        metrics = self._get_abc_metrics(self._current_circuit)
        if metrics.get("success"):
            self._last_metrics = metrics
            # Capture initial LUT6 and Level for this sequence
            self.lut_init = metrics.get("lut6_count")
            self.lev_init = metrics.get("lut6_lev")
        self._step_count = 0

        info = {
            "score": self._score_state(self._state),
            "action_names": self._enabled_action_names,
            "circuit": str(self._current_circuit),
            "base_circuit": str(self._active_base_circuit),
            "mode": self.mode,
            "metrics": metrics,
            "state_file": str(self._current_state_file),
            "lut_init": self.lut_init,
            "lev_init": self.lev_init,
        }
        return self._state.copy(), info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}; expected [0, {self.num_actions - 1}]")

        state_file_before = str(self._current_state_file)
        prev_metrics = self._get_abc_metrics(self._current_circuit)

        action_name = self._enabled_action_names[action]
        if action_name in MIG_ACTION_NAMES:
            exec_info = self._run_mig_action(action_name)
        else:
            exec_info = self._run_abc_action(action_name)

        reencode_info: Optional[Dict[str, Any]] = None
        clean_signal_breach = False
        if exec_info.get("success") and self.enable_real_state_reencode:
            reencode_info = self._reencode_state_from_circuit(self._current_circuit, step_index=self._step_count + 1)
            if reencode_info.get("success"):
                next_state = reencode_info["state"]
                self._current_state_file = Path(reencode_info["state_file"])
                next_score = self._score_state(next_state)
            else:
                clean_signal_breach = True
                if self.enforce_clean_signal:
                    next_state = self._state.copy()
                    next_score = self._score_state(next_state)
                else:
                    next_state = self._apply_action(self._state, action)
                    next_score = self._score_state(next_state)
        else:
            clean_signal_breach = True
            if self.enforce_clean_signal:
                next_state = self._state.copy()
                next_score = self._score_state(next_state)
            else:
                next_state = self._apply_action(self._state, action)
                next_score = self._score_state(next_state)

        next_metrics = self._get_abc_metrics(self._current_circuit)
        if not next_metrics.get("success"):
            clean_signal_breach = True

        # Calculate QoR using unified LUT6/Level formula: alpha * ((lut_init - lut_t) / lut_init) + beta * ((lev_init - lev_t) / lev_init)
        next_lut6 = next_metrics.get("lut6_count")
        next_lev = next_metrics.get("lut6_lev")
        
        if (self.enable_real_qor_reward and next_lut6 is not None and next_lev is not None 
            and self.lut_init is not None and self.lev_init is not None
            and self.lut_init != 0 and self.lev_init != 0):
            # New unified formula: relative improvement in LUT6 and Level
            lut_improvement = (self.lut_init - next_lut6) / self.lut_init
            lev_improvement = (self.lev_init - next_lev) / self.lev_init
            qor = float(self.reward_alpha * lut_improvement + self.reward_beta * lev_improvement)
        else:
            # Fallback to surrogate state-based reward
            prev_score = self._score_state(self._state)
            qor = float((next_score - prev_score))

        # Reward is QoR minus optional step penalty
        # Reward is QoR minus optional step penalty.
        reward_raw = float(qor - self.step_penalty)
        if clean_signal_breach and self.enforce_clean_signal:
            reward_raw = min(reward_raw, -0.01)

        # Normalize and clip reward to keep PPO updates numerically stable.
        reward = reward_raw / max(self.reward_scale, 1e-8)
        reward = float(np.clip(reward, -self.reward_clip, self.reward_clip))

        self._state = next_state
        self._step_count += 1

        terminated = bool(clean_signal_breach and self.enforce_clean_signal)
        truncated = self._step_count >= self.max_steps

        info = {
            "score": next_score,
            "action_id": int(action),
            "action_name": action_name,
            "allowed_action_names": self._enabled_action_names,
            "step": self._step_count,
            "circuit": str(self._current_circuit),
            "base_circuit": str(self._active_base_circuit),
            "mode": self.mode,
            "action_exec": exec_info,
            "metrics_prev": prev_metrics,
            "metrics_next": next_metrics,
            "qor_next": qor,
            "reward_raw": reward_raw,
            "reward_scaled": reward,
            "reward_scale": self.reward_scale,
            "reward_clip": self.reward_clip,
            "reward_mode": "lut6_level_relative" if (self.enable_real_qor_reward and next_lut6 is not None and next_lev is not None) else "surrogate",
            "state_reencode": reencode_info,
            "clean_signal_breach": clean_signal_breach,
            "clean_signal_enforced": self.enforce_clean_signal,
            "state_file_before": state_file_before,
            "state_file": str(self._current_state_file),
            "lut_init": self.lut_init,
            "lev_init": self.lev_init,
        }
        return self._state.copy(), reward, terminated, truncated, info

    def render(self) -> None:
        return None

    def close(self) -> None:
        return None
