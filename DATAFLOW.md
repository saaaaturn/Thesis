# HybridSYN Data Flow Architecture

## Overview: Complete Data Journey

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        INITIALIZATION PHASE                             │
└─────────────────────────────────────────────────────────────────────────┘

    AIGER Files (Benchmarks)
           │
           ├─ arithmetic/adder.aig          (256 inputs, 129 outputs)
           ├─ arithmetic/multiplier.aig     (128 inputs, 128 outputs)
           ├─ random_control/arbiter.aig    (256 inputs, 129 outputs)
           └─ ... (23 total circuits)
           │
           ▼
    ┌──────────────────────────────────┐
    │ import.cpp (C++ Conversion)      │
    │ ├─ Read AIGER file               │
    │ ├─ Load into AIG network          │
    │ └─ Load into MIG network          │
    └──────────────────────────────────┘
           │
           ├─ AIG Structure (in RAM)       MIG Structure (in RAM)
           │  └─ Nodes/Gates                └─ Nodes/Gates
           │  └─ Edges/Connections          └─ Edges/Connections
           │  └─ Root signals               └─ Root signals
           │  └─ PPI/PPO                    └─ PPI/PPO
           │
           ▼ ▼
    ┌────────────────────────────────────────────────────┐
    │ Export to Python (via interface.cpp or JSON)       │
    │ ├─ AIG topology (adjacency list)                   │
    │ ├─ MIG topology (adjacency list)                   │
    │ ├─ Node/gate count                                 │
    │ ├─ Logic depth                                     │
    │ ├─ I/O information                                 │
    │ └─ Statistical features                            │
    └────────────────────────────────────────────────────┘
           │
           ▼

┌─────────────────────────────────────────────────────────────────────────┐
│                    FEATURE EXTRACTION PHASE                             │
│                        (Python: PyTorch)                                │
└─────────────────────────────────────────────────────────────────────────┘

    AIG Graph                          MIG Graph
    (nodes, edges)                     (nodes, edges)
           │                                  │
           ▼                                  ▼
    ┌──────────────────┐            ┌──────────────────┐
    │  GCN Layer 1     │            │  GCN Layer 1     │
    │ (node embedding) │            │ (node embedding) │
    │  + BatchNorm     │            │  + BatchNorm     │
    │  + ReLU          │            │  + ReLU          │
    └────────┬─────────┘            └────────┬─────────┘
             │                               │
             ▼                               ▼
    ┌──────────────────┐            ┌──────────────────┐
    │  GCN Layer 2     │            │  GCN Layer 2     │
    │ (node embedding) │            │ (node embedding) │
    │  + BatchNorm     │            │  + BatchNorm     │
    │  + ReLU          │            │  + ReLU          │
    └────────┬─────────┘            └────────┬─────────┘
             │                               │
             ▼                               ▼
    ┌──────────────────┐            ┌──────────────────┐
    │   Mean Pooling   │            │   Mean Pooling   │
    │   Max Pooling    │            │   Max Pooling    │
    │                  │            │                  │
    │  Concatenate     │            │  Concatenate     │
    │  → AIG Features  │            │  → MIG Features  │
    │   (vector size:) │            │   (vector size:) │
    │   e.g., 128D     │            │   e.g., 128D     │
    └──────────┬───────┘            └──────────┬───────┘
               │                               │
               └───────────────┬───────────────┘
                               │
                               ▼
                ┌─────────────────────────────────┐
                │   Statistical Features          │
                ├─────────────────────────────────┤
                │ • LUT-6 count (after mapping)   │
                │ • Logic depth                   │
                │ • Input/Output count            │
                │ • Gate distribution             │
                │ • Fanout/Fanin statistics       │
                │ • Node type histogram           │
                │                                 │
                │ Feature vector: (e.g., 10D)    │
                └──────────┬──────────────────────┘
                           │
                           ▼
                ┌─────────────────────────────────┐
                │   Cross-Attention Fusion        │
                │                                 │
                │  Input: AIG_features [128D]     │
                │          MIG_features [128D]    │
                │          Stats [10D]            │
                │                                 │
                │  Mechanism:                     │
                │   Attention = softmax(Q, K, V)  │
                │   ├─ Query: Stats features      │
                │   ├─ Key: AIG + MIG features    │
                │   └─ Value: weighted sum        │
                │                                 │
                │  Output: Fused_features [256D]  │
                └──────────┬──────────────────────┘
                           │
                           ▼
                ┌─────────────────────────────────┐
                │  Fully Connected Layer          │
                │ (Shared Latent Representation)  │
                │                                 │
                │  Input: Fused_features [256D]   │
                │  Output: latent_state [128D]    │
                └──────────┬──────────────────────┘
                           │
                           ▼

┌─────────────────────────────────────────────────────────────────────────┐
│                  REINFORCEMENT LEARNING PHASE                           │
│              (Gymnasium Environment + Stable-Baselines3)                │
└─────────────────────────────────────────────────────────────────────────┘

                  Latent State [128D]
                           │
                ┌──────────┴────────────┐
                │                       │
                ▼                       ▼
        ┌─────────────────┐    ┌─────────────────┐
        │ Policy Network  │    │ Value Network   │
        │                 │    │                 │
        │  FC: 128 → 64   │    │  FC: 128 → 64   │
        │  ReLU           │    │  ReLU           │
        │  FC: 64 → 32    │    │  FC: 64 → 32    │
        │  ReLU           │    │  ReLU           │
        │  FC: 32 → 12    │    │  FC: 32 → 1     │
        │  Softmax        │    │ (scalar)        │
        │                 │    │                 │
        │  Output:        │    │  Output:        │
        │  Action Probs   │    │  State Value    │
        │  [12D]          │    │  [1D]           │
        └────────┬────────┘    └────────┬────────┘
                 │                      │
                 │      Predicted Value │
                 │        (for loss)    │
                 │                      │
                 ▼
        ┌─────────────────┐
        │ Action Sampling │
        │ (or argmax)     │
        │                 │
        │ Choose action:  │
        │ • 0-5: ABC ops  │
        │ • 6-11: MIG ops │
        │                 │
        │ Sampled_action  │
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────────────────────────────┐
        │  Environment Step (interface.cpp)       │
        │  ┌─────────────────────────────────────┤
        │  │ Input: Selected operator             │
        │  │ Action: Apply to AIG or MIG          │
        │  │  ├─ ABC operator (if 0-5)            │
        │  │  │  └─ e.g., "rewrite -l"           │
        │  │  └─ MIG operator (if 6-11)           │
        │  │     └─ e.g., "refactor"              │
        │  │                                      │
        │  │ Output: New circuit state            │
        │  │  ├─ Modified AIG/MIG                │
        │  │  ├─ New LUT count                    │
        │  │  ├─ New logic depth                  │
        │  │  └─ New gate count                   │
        │  └─────────────────────────────────────┤
        │                                         │
        │ Reward Calculation:                     │
        │  r_t = -(α × LUT_count                  │
        │         + β × |logic_depth|)            │
        │                                         │
        │  Example:                               │
        │   LUT_count_before: 254                 │
        │   LUT_count_after:  240   (reduced!)    │
        │   logic_depth_before/after: stable      │
        │   Reward: r_t = -240 (better = lower)   │
        └─────────────────────────────────────────┘
                 │
                 ├─ New State: new circuit features
                 ├─ Reward: r_t (scalar)
                 ├─ Terminated: episodic flag
                 └─ Info: metadata

                 ▼
        ┌─────────────────────────────┐
        │  Store in Replay Buffer     │
        │  ┌───────────────────────────┤
        │  │ Transition:               │
        │  │ (s_t,                     │
        │  │  a_t,                     │
        │  │  r_t,                     │
        │  │  s_t+1,                   │
        │  │  done,                    │
        │  │  log_prob,                │
        │  │  value_estimate)          │
        │  └───────────────────────────┤
        │                              │
        │ Buffer stores last           │
        │ N transitions for PPO        │
        │ minibatch updates            │
        └──────────┬───────────────────┘
                   │
                   ▼
        ┌─────────────────────────────┐
        │   PPO Update Phase          │
        │  ┌───────────────────────────┤
        │  │ When buffer fills:        │
        │  │                           │
        │  │ 1. Compute GAE:           │
        │  │    Ã_t = Σγ^l δ_t+l      │
        │  │                           │
        │  │ 2. Policy Loss:           │
        │  │    L^clip = -Ê_t[        │
        │  │      min(r_t Â_t,        │
        │  │      clip(r_t) Â_t)]     │
        │  │                           │
        │  │ 3. Value Loss:            │
        │  │    L^v = E[(V(s) - v_t)²] │
        │  │                           │
        │  │ 4. Entropy bonus:         │
        │  │    L^ent = -α H(π)        │
        │  │                           │
        │  │ Total Loss:               │
        │  │ L^total = L^clip +        │
        │  │           c1×L^v +        │
        │  │           c2×L^ent        │
        │  │                           │
        │  │ 5. SGD update on          │
        │  │    networks               │
        │  └───────────────────────────┤
        │                              │
        │ Repeat for K epochs over     │
        │ minibatches from buffer      │
        └──────────┬───────────────────┘
                   │
                   ▼
        ┌─────────────────────────────┐
        │   Next Episode Begins       │
        │   Reset environment with    │
        │   different circuit         │
        │   Repeat N_episodes times   │
        └─────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                      OUTPUT/EVALUATION PHASE                            │
└─────────────────────────────────────────────────────────────────────────┘

    Trained PPO Model
           │
           ├─ Policy network weights
           ├─ Value network weights
           └─ Saved checkpoint
           │
           ▼
    ┌─────────────────────────────────────────┐
    │  Inference on Test Circuits             │
    │  ┌─────────────────────────────────────┤
    │  │ For each test circuit:                │
    │  │                                      │
    │  │ 1. Extract features (GCN)            │
    │  │ 2. Forward through policy            │
    │  │ 3. Sample actions greedily           │
    │  │    (no exploration)                  │
    │  │ 4. Apply operators                   │
    │  │ 5. Measure final:                    │
    │  │    • LUT count reduction             │
    │  │    • Logic level reduction           │
    │  │    • Synthesis time                  │
    │  └─────────────────────────────────────┤
    │                                         │
    │ Results:                                │
    │  • 9.7% better LUT than RL-A2C          │
    │  • 9.7% better LUT than RL-PPO (AIG)   │
    │  • 17.1% better LUT than resyn2 heur.  │
    │  • 46.8 LUTs vs 23.9% shallower depth  │
    └─────────────────────────────────────────┘
           │
           ├─ Optimized circuits
           ├─ Metrics (CSV)
           ├─ Comparison plots
           └─ Training logs
           │
           ▼
    3_outputs/
    ├─ logs/PPO_test/
    │  └─ events, scalars (TensorBoard)
    └─ models/PPO_test/
       └─ model.zip (trained weights)
```

---

## Detailed Stage-by-Stage Breakdown

### STAGE 1: Circuit Loading (import.cpp)

**Input**: AIGER benchmark files
```
File: 1_inputs/EPFL_benchmarks/arithmetic/adder.aig
Format: 
  - Header: aig <max_var_idx> <num_inputs> <num_latches> <num_outputs> <num_ands>
  - Gates: binary format with literal encodings
```

**Process**:
```cpp
aig_network aig;           // Reserve AIG structure in RAM
mig_network mig;           // Reserve MIG structure in RAM

read_aiger(filename, aiger_reader(aig));  // Parse & populate AIG
read_aiger(filename, aiger_reader(mig));  // Parse & populate MIG
```

**Output**: Two in-memory graph representations
```
AIG State Example:
├─ Nodes: {AND1, AND2, ..., AND_254}
├─ Edges: connections between gates
├─ Gate count: 254
├─ Logic depth: 51
└─ I/O: 256 inputs, 129 outputs

MIG State Example:
├─ Nodes: {MAJ1, MAJ2, ..., MAJ_M}
├─ Edges: connections between gates
├─ Gate count: potentially different
├─ Logic depth: potentially different
└─ I/O: 256 inputs, 129 outputs
```

---

### STAGE 2: Feature Extraction (GCN Processing)

**Input**: AIG and MIG graph structures

**Graph Convolutional Network Processing**:

```
For AIG:
  node_features[i] = f(node_i, neighbors_of_i)
  
  Initial: node_type, fanin_count, fanout_count
  
  GCN1: aggregate neighbor embeddings
        message = (neighbor_embeddings · W_agg)
        embedding = ReLU(embedding + message)
  
  GCN2: deeper aggregation
        embedding = ReLU(embedding + aggregate_neighbors())
  
  Pooling: global_representation
        mean_pool = mean(all_node_embeddings)
        max_pool = max(all_node_embeddings)
        final_aig_features = concat(mean_pool, max_pool)
```

**Statistical Features**:
```
Features extracted:
├─ Circuit complexity
│  ├─ Number of AND/MAJORITY gates
│  ├─ Logic depth (longest path)
│  ├─ Number of primary inputs
│  └─ Number of primary outputs
├─ Gate distribution
│  ├─ AND gates: percentage
│  ├─ NOT gates: percentage
│  └─ Other: percentage
└─ Connectivity
   ├─ Average fanin
   ├─ Average fanout
   ├─ Critical path length
   └─ Gate density
```

**Output**:
```
For each circuit representation:
╔════════════════════════════════════════╗
║  AIG_features: [128D]                  ║
║  MIG_features: [128D]                  ║
║  Statistical_features: [10D]           ║
╚════════════════════════════════════════╝
```

---

### STAGE 3: Cross-Attention Fusion

**Purpose**: Learn which representation is more important for CURRENT circuit

**Input**: 
- AIG_features [128D]
- MIG_features [128D]  
- Statistical_features [10D]

**Process**:
```
Query = FC1(Statistical_features)     # What we're asking
Key = FC2(concat(AIG, MIG))           # What's available
Value = concat(AIG_features, MIG_features)  # What we use

Attention_weights = softmax(Query · Key^T)
Weighted_features = Attention_weights · Value

Example: If circuit is arithmetic-heavy:
  Attention ≈ [0.3, 0.7]  # Value MIG more
  
If circuit is control-heavy:
  Attention ≈ [0.7, 0.3]  # Value AIG more
```

**Output**: 
```
Fused_features [256D] = concatenate(
  Attention_weighted_AIG,
  Attention_weighted_MIG
)
```

---

### STAGE 4: Policy Network Decision

**Input**: Fused_features [256D]

**Network**:
```
Dense(256 → 64) + ReLU
Dense(64 → 32) + ReLU
Dense(32 → 12) + Softmax

Output: [p_0, p_1, ..., p_11]
  where p_i = probability of action i
```

**Action Space** (12 discrete actions):
```
Actions 0-5 (ABC operators):
├─ 0: rewrite -l     (level-driven rewrite)
├─ 1: refactor -l    (level-driven refactor)
├─ 2: balance        (balanced tree)
├─ 3: resub -l       (resubstitution)
├─ 4: ps             (print structure)
└─ 5: (varies)

Actions 6-11 (MIG/Mockturtle operators):
├─ 6: refactor       (MIG refactor)
├─ 7: balance        (MIG balance)
├─ 8: resub          (MIG resubstitution)
├─ 9: optimize       (MIG optimize)
├─ 10: zero_cost     (zero-cost optimization)
└─ 11: (varies)
```

**Sampling**:
```
action = categorical_sample(p_0, p_1, ..., p_11)
# OR for greedy: action = argmax(p_i)
```

---

### STAGE 5: Environment Step & Reward

**Input**: Selected action

**Execution**:
```
if action in [0-5]:
  call_ABC_operator(action, aig_circuit)
  new_aig = result
  new_mig = convert_aig_to_mig(new_aig)
else:
  call_MIG_operator(action-6, mig_circuit)
  new_mig = result
  new_aig = convert_mig_to_aig(new_mig)

# Map to FPGA LUTs (e.g., using ABC "if" command)
lut_mapping = abc_map_to_luts(new_circuit)
new_lut_count = count_lutval
new_logic_depth = measure_depth(new_circuit)
```

**Reward Calculation**:
```
reward = -(α × LUT_count + β × |logic_depth|)

Typical values:
α = 1.0  (optimize for area)
β = 0.1  (maintain reasonable depth)

Example:
Before: LUT=254, Depth=51
After:  LUT=240, Depth=50
Reward_before = -(1.0×254 + 0.1×51) = -259.1
Reward_after  = -(1.0×240 + 0.1×50) = -245.0
Step_reward   = -245.0 - (-259.1) = +14.1 ✓ (improvement!)
```

**Return Value**:
```
(new_state, reward, terminated, info) = env.step(action)

new_state: [128D]  # computed from new circuit
reward: float       # scalar reward value
terminated: bool    # episode finished?
info: dict         # metadata
```

---

### STAGE 6: PPO Learning Update

**Buffer Transition Storage**:
```
transition = {
  'observation': state_t,
  'action': a_t,
  'reward': r_t,
  'next_observation': state_t+1,
  'done': terminated,
  'log_probability': log(π(a_t|s_t)),
  'value_estimate': V(s_t)
}
```

**GAE (Generalized Advantage Estimation)**:
```
δ_t = r_t + γ V(s_t+1) - V(s_t)  # advantage
Â_t = Σ (λγ)^l δ_t+l             # smoothed advantage
```

**Policy Gradient Loss**:
```
L^clip = -E_t[
  min(
    r_t * Â_t,
    clip(r_t, 1-ε, 1+ε) * Â_t
  )
]

# Prevents too-large policy updates
```

**Value Loss**:
```
L^value = E_t[(V(s_t) - target_value)²]

# target_value = r_t + γ V(s_t+1)
```

**Total Loss**:
```
L_total = L^clip + c1 × L^value + c2 × H(π)

where H(π) = entropy for exploration
c1, c2 = hyperparameters (0.5, 0.01 typical)
```

**Update**:
```
θ ← θ - ∇_θ L_total  (SGD/Adam optimizer)
```

---

### STAGE 7: Episode and Training Loop

**Episode Structure**:
```
Episode 1:
├─ Reset: Select random circuit from EPFL
├─ Step 1: observe s_0 → take a_0 → reward r_0
├─ Step 2: observe s_1 → take a_1 → reward r_1
├─ Step 3: observe s_2 → take a_2 → reward r_2
├─ ...
└─ Step L: done (reach max episode length L=25)

Training Loop:
for episode in range(200):  # 200 total episodes
  circuit = sample_random_circuit()
  for step in range(25):    # 25 steps per episode
    state = get_features(circuit)
    action = policy.sample(state)
    state', reward = env.step(action)
    buffer.add(state, action, reward, state')
    
  if (episode % K == 0) and buffer.is_full():
    for epoch in range(PPO_epochs):  # multiple passes
      minibatch = buffer.sample()
      loss = compute_loss(minibatch)
      optimizer.step(loss)
```

---

## Data Flow Summary Table

| Stage | Input Type | Component | Output Type | Size |
|-------|-----------|-----------|-------------|------|
| 1 | File (AIGER) | import.cpp | Graph (AIG/MIG) | Variable |
| 2 | Graph (AIG/MIG) | GCN + Pool | Feature vectors | 128D + 128D + 10D |
| 3 | Features | Cross-Attention | Fused features | 256D |
| 4 | Fused features | Policy Network | Action probabilities | 12D |
| 5 | Action ID | Environment | (state, reward) | 128D + scalar |
| 6 | Transition tuple | PPO Loss | Network gradients | - |
| 7 | Gradients | Optimizer | Updated networks | - |

---

## Key Information Transformations

```
Circuit File (KB)
    ↓ [Parsing]
Graph Structure (Nodes, Edges, Gate types)
    ↓ [GCN Embedding]
Node embeddings (128D per node) + Statistics
    ↓ [Pooling & Attention]
Single Circuit Representation (256D)
    ↓ [Policy Network]
Action Distribution (12 logits)
    ↓ [Sampling]
Operator Choice (1 action)
    ↓ [Execution]
Modified Circuit + Metrics
    ↓ [Evaluation]
Reward Signal (scalar) + New State
    ↓ [RL Learning]
Updated Policy Parameters
```

