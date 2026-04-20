# HybridSYN Handover for Google Antigravity

## 1. Mission
HybridSYN trains a PPO agent to select synthesis actions (ABC + MIG) that improve mapped FPGA QoR. The active reward definition is LUT6/Level relative improvement, and all key scripts now use consistent reward arguments.

## 2. Current Working Status (Verified)
1. End-to-end workflow is validated for train -> continue -> inference.
2. Both single mode and multiple mode are running correctly.
3. QoR reward uses unified LUT6 count + LUT6 mapped level formula.
4. Step penalty default is 0.0 for clean dense reward signal.
5. Initial per-sequence LUT6 baseline is captured in reset and reused through the episode.
6. Resyn2 baseline runner is implemented and tested.

## 3. Active Reward and QoR Definition
Current QoR at step t:

QoR_t = alpha * ((LUT_init - LUT_t) / LUT_init) + beta * ((LEV_init - LEV_t) / LEV_init)

Current reward:

Reward_t = QoR_t - step_penalty

Important details:
1. LUT_t and LEV_t are from ABC mapping with if -K 6 (LUT6 metrics).
2. LEV_t is LUT6 mapped level, not raw AIG depth.
3. Default training values:
- reward_alpha = 0.5
- reward_beta = 0.5
- step_penalty = 0.0
- reward_scale = 50.0
- reward_clip = 10.0

## 4. Pipeline Overview
1. Load input AIG circuit.
2. Run chosen action (ABC or MIG) on current circuit.
3. Re-extract graph features and regenerate 320D state embedding.
4. Compute LUT6 metrics via ABC if -K 6 for QoR/reward.
5. PPO update uses rollout data (gamma + gae_lambda configured in train script).

Main files:
- 2_process/hybridsyn_env.py: env logic, action execution, reward, metric collection.
- 2_process/train_hybridsyn_ppo.py: initial training entrypoint.
- 2_process/continue_training.py: resume training from checkpoint/final model.
- 2_process/run_trained_model.py: inference/evaluation with detailed traces.
- 2_process/run_resyn2_baseline.py: fixed-sequence ABC baseline (resyn2 alias).

## 5. Modes and Artifact Layout
1. single mode:
- one fixed circuit for the whole run.
- artifacts under 3_outputs/<circuit_name>/<run_name>/...

2. multiple mode:
- cycle through circuit pool across resets.
- artifacts under 3_outputs/MULTIPLE_MODE/<run_name>/...

Common artifacts:
1. run log and run_args.json
2. final model zip
3. checkpoint zips in checkpoints/
4. optional eval outputs in best/ and eval/

## 6. Logging and Metric Semantics
Training step line contains both:
1. Depth = raw AIG level (from strash; print_stats)
2. LUT6_Level = mapped level after if -K 6

For benchmark/paper comparison with LUT-6 tables, use:
1. LUT6 count
2. LUT6_Level

Do not compare benchmark LUT6 levels against raw Depth.

## 7. Baseline Comparison Contract (Resyn2)
Baseline script:
- 2_process/run_resyn2_baseline.py

Behavior:
1. sources abc.rc so resyn2 alias exists.
2. runs read_aiger; strash; resyn2; if -K 6; print_stats.
3. writes CSV with circuit, success, lut6_count, lut6_lev, tails for debug.

Fair comparison rules:
1. same circuit set for HybridSYN and baseline.
2. same mapping step if -K 6.
3. compare LUT6 count with LUT6 count, LUT6_Level with Level.

## 8. Known Fixes Already Applied
1. Fixed reset fallback_state scope bug in hybridsyn_env.py.
2. Added consistent reward args across train/continue/inference scripts.
3. Added per-circuit evaluation behavior for multiple mode callback path.
4. Baseline parser now reads final mapped nd/lev occurrence correctly.

## 9. Operational Pitfalls
1. If total-actions < eval-freq, eval artifacts may be missing or minimal.
2. Very frequent save/eval increases runtime overhead.
3. Confusing Depth with LUT6_Level leads to wrong conclusions.
4. Running ABC without sourcing abc.rc may break resyn2 alias command.

## 10. Recommended Commands (Condensed)
From 2_process directory:

Training (single):
```bash
python train_hybridsyn_ppo.py \
  --mode single \
  --circuit-file ../1_inputs/EPFL_benchmarks/arithmetic/adder.aig \
  --sequence-steps 25 \
  --total-actions 30000 \
  --ppo-steps 300 \
  --reward-alpha 0.5 \
  --reward-beta 0.5 \
  --step-penalty 0.0 \
  --log-file-name train_single_lutlevel.log
```

Continue:
```bash
python continue_training.py \
  --model-path ../3_outputs/adder/train_single_lutlevel/ppo_hybridsyn_final.zip \
  --additional-actions 30000
```

Inference:
```bash
python run_trained_model.py \
  --model-path ../3_outputs/adder/train_single_lutlevel/ppo_hybridsyn_final.zip \
  --mode single \
  --circuit-file ../1_inputs/EPFL_benchmarks/arithmetic/adder.aig \
  --episodes 3
```

Resyn2 baseline:
```bash
python run_resyn2_baseline.py \
  --input-glob ../1_inputs/EPFL_benchmarks/arithmetic/*.aig \
  --abc-binary abc/abc \
  --abc-rc abc/abc.rc \
  --output-csv ../3_outputs/resyn2_baseline.csv
```

## 11. Recommended Antigravity Onboarding Prompt
"You are joining the HybridSYN PPO codebase. First restate reward semantics, metric semantics (Depth vs LUT6_Level), mode-specific artifact layout, and baseline comparison contract from HANDOVER_ANTIGRAVITY.md. Then list likely regression risks if modifying reward, reset logic, logging, and mapping commands. Finally propose a minimal change plan with concrete acceptance checks (files, log lines, and numeric metric sanity checks)."

## 12. Immediate Next Tasks
1. Generate unified comparison table HybridSYN vs resyn2 (all arithmetic circuits).
2. Add explicit labels in logs (AIG_Depth vs LUT6_Level) to prevent metric confusion.
3. Add automated validation script that checks reward mode, metric fields, and expected artifacts after a short run.
