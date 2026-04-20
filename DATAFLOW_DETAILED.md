# HybridSYN Data Flow - Complete Analysis

## Executive Summary

HybridSYN is a 7-stage data pipeline that transforms circuit files into optimized netlists using reinforcement learning:

```
AIGER Files → Parse → Feature Extract → Fuse → RL Policy → Optimize → Evaluate
```

Each stage handles specific data transformations, with clear input/output specifications.

---

## Stage 1: Circuit Loading & Parsing

### Input
- **Format**: AIGER (And-Inverter Graph in Easy format)
- **Source**: EPFL benchmark cirsuit files
- **Size**: 10-100 KB per file
- **Examples**: `adder.aig`, `multiplier.aig`, `arbiter.aig`

### Processing (import.cpp)
```cpp
// Dual representation loading
aig_network aig;
mig_network mig;

read_aiger(filename, aiger_reader(aig));  // Parse into AIG
read_aiger(filename, aiger_reader(mig));  // Parse into MIG (auto-converted)
```

### Output
Two graph structures representing the same circuit:

**AIG Structure**:
- Nodes: AND gates + primary inputs/outputs
- Edges: Signal connections
- Gate count: ~100-10,000 nodes (circuit dependent)
- Typical: ~254 gates for 256-bit adder

**MIG Structure**: 
- Nodes: Majority gates + primary inputs/outputs
- Edges: Signal connections
- Gate count: Usually different from AIG
- May be more or fewer gates (rep. dependent)

---

## Stage 2: Feature Extraction via Graph Convolutional Networks

### Input
Two graph structures (AIG & MIG) with node features:
- Node type (gate type: AND, OR, NOT, MAJ, etc.)
- Connectivity information (adjacency list)
- Signal fanin/fanout counts

### Processing

**GCN for AIG**:
```
For each node in graph:
  neighbors_info = aggregate_neighbors()
  node_embedding = ReLU(GCN1_weights @ neighbors_info)

For all nodes again:
  refined_embedding = ReLU(GCN2_weights @ neighbors_info)
```

**GCN for MIG**: Same process with separate weights

**Output of GCN**:
- AIG: Node embeddings [N × 128D] where N = number of AND gates
- MIG: Node embeddings [K × 128D] where K = number of majority gates

### Pooling (Aggregate to Circuit Level)

```
Global avg pool:    mean(all_node_embeddings) = [128D]
Global max pool:    max(all_node_embeddings) = [128D]
Concatenate:        [128D] + [128D] = [256D]

But we want [128D], so reduce:
Circuit representation = FC_layer([256D]) = [128D]
```

**Final output from Feature Extraction**:
- AIG features: [128D]
- MIG features: [128D]

### Statistical Features (Extracted Separately)

```
Circuit metrics:
├─ Gate count (overall)
├─ Logic depth (longest path length)
├─ Number of primary inputs  
├─ Number of primary outputs
├─ Gate type distribution (% AND, % NOT, etc.)
├─ Average fanin per node
├─ Average fanout per node
├─ Feedback loop indicators
├─ Critical path length
└─ Gate density

Total: ~10D feature vector
```

---

## Stage 3: Cross-Attention Fusion

### Input
- AIG features: [128D]
- MIG features: [128D]
- Statistical features: [10D]

### Why This Stage Exists

The key insight: **Different circuits optimize better with different representations**

- Arithmetic circuits → MIG works better
- Control circuits → AIG works better
- Mixed circuits → need both

**Cross-attention learns**: "For THIS circuit, how much should I care about AIG vs MIG?"

### Processing

```python
# Three components
query = FC_layer(statistical_features)      # What to ask [128D]
key = FC_layer(concat(aig_feat, mig_feat))  # What's available [256D]
value = concat(aig_feat, mig_feat)          # What to use [256D]

# Attention mechanism
weights = softmax(query @ key^T / sqrt(d_k))  # [2] shape

# Weighted combination
fused = weights[0] * aig_feat + weights[1] * mig_feat  # [128D]
```

### Example Outputs

**For an arithmetic circuit** (like multiplier):
```
weights ≈ [0.3, 0.7]  # Value MIG more (70%)
Learned: "MIG is better for arithmetic, use 70% MIG info"
```

**For a control circuit** (like arbiter):
```
weights ≈ [0.7, 0.3]  # Value AIG more (70%)
Learned: "AIG is better for control, use 70% AIG info"
```

### Output
- Fused representation: [256D]
  - Contains weighted information from both representations
  - Adapted to circuit characteristics

---

## Stage 4: Neural Network Policy & Value Heads

### Input
- Fused features: [256D]

### Architecture

**Shared Latent Layer**:
```
FC: 256D → 128D (ReLU)
Output: shared_latent [128D]
```

**Policy Head** (actor):
```
FC: 128D → 64D (ReLU)
FC: 64D → 32D (ReLU)
FC: 32D → 12D (Softmax)
Output: action_logits [12D]
  Interpretation: probability for each of 12 actions
```

**Value Head** (critic):
```
FC: 128D → 64D (ReLU)
FC: 64D → 32D (ReLU)
FC: 32D → 1D (identity)
Output: value_estimate [1D]
  Interpretation: expected cumulative reward from this state
```

### Outputs

**From Policy Head**:
```python
action_probs = softmax(action_logits)  # [p_0, p_1, ..., p_11]
# p_i = probability of selecting action i
# Σp_i = 1.0

# To select action:
action = sample(action_probs)  # During training (exploration)
# OR
action = argmax(action_probs)  # During inference (exploitation)
```

**From Value Head**:
```python
value_estimate = scalar  # Single floating-point number
# Interpretation: "From this state, I expect to get value V"
# Used for advantage estimation: advantage = reward + γ*V(next) - V(current)
```

---

## Stage 5: Environment Execution & Reward

### Input
- Selected action: integer in [0, 11]
- Current circuit: AIG or MIG structure

### Action Mapping

**Actions 0-5: ABC Operators** (via subprocess or library call)
```
action 0: abc -c "rewrite -l"      (level-driven rewrite)
action 1: abc -c "refactor -l"     (level-driven refactor)
action 2: abc -c "balance"         (balance tree)
action 3: abc -c "resub -l"        (resubstitution)
action 4: abc -c "ps"              (print structure)
action 5: abc -c "custom_op"       (custom operation)
```

**Actions 6-11: MIG Operators** (via mockturtle)
```
action 6: mig.refactor()
action 7: mig.balance()
action 8: mig.resub()
action 9: mig.optimize()
action 10: mig.zero_cost_rewrite()
action 11: mig.custom_op()
```

### Processing

```cpp
// Apply operator to circuit
if (action < 6) {
  aig = apply_abc_operator(aig, action);
  mig = convert_aig_to_mig(aig);  // Keep MIG in sync
} else {
  mig = apply_mig_operator(mig, action - 6);
  aig = convert_mig_to_aig(mig);  // Keep AIG in sync
}

// Evaluate new circuit
int new_lut_count = abc_map_to_fpga_luts(circuit, 6);  // Map to 6-LUTs
int new_depth = measure_logic_depth(circuit);
```

### Reward Calculation

**Formula**:
```
r_t = -(α × LUT_count + β × |logic_depth|)

Hyperparameters:
α = 1.0        (primary objective: minimize area)
β = 0.1        (secondary: keep depth reasonable)

Weights indicate: "LUT count is 10x more important than depth"
```

**Concrete Example**:
```
Initial circuit:
  LUT_count = 254
  depth = 51
  r_0 = -(1.0 × 254 + 0.1 × 51) = -(254 + 5.1) = -259.1

After action 1 (refactor -l):
  LUT_count = 240  (reduced by 14!)
  depth = 50       (improved!)
  r_1 = -(1.0 × 240 + 0.1 × 50) = -(240 + 5.0) = -245.0

Step reward = r_1 - r_0 = -245.0 - (-259.1) = +14.1
Interpretation: "Good move! Reduced LUTs by 14"
```

### Return Value

Environment step returns 4-tuple:
```python
(new_state, reward, terminated, info) = env.step(action)

new_state: [256D]         # Features of modified circuit
reward: float             # Scalar reward value
terminated: bool          # True if episode should end
info: dict                # {"lut_count": 240, "depth": 50, ...}
```

---

## Stage 6: Replay Buffer & PPO Learning

### Transition Storage

Each environment step generates a transition tuple:
```python
transition = {
    'observation': state_t,                    # [256D]
    'action': a_t,                             # scalar: 0-11
    'reward': r_t,                             # scalar
    'next_observation': state_{t+1},           # [256D]
    'done': terminated,                        # bool
    'log_probability': log(π(a_t | s_t)),     # scalar (for policy)
    'value_estimate': V(s_t),                 # scalar (for value)
}
```

### Replay Buffer

```python
buffer = []  # List of transitions
max_size = 2048  # Store up to 2048 transitions

# After each step:
buffer.append(transition)

# When buffer fills:
if len(buffer) >= max_size:
    perform_ppo_update()
    buffer.clear()  # Reset for next batch
```

### GAE (Generalized Advantage Estimation)

```python
# For each transition in batch
temporal_difference = reward_t + γ * V(s_{t+1}) - V(s_t)

# Accumulate with discount factor λ
advantage = sum(
    (λ × γ)^l × temporal_difference_{t+l}
    for l in range(horizon)
)

# This gives "how much better/worse than expected"
```

### PPO Loss Functions

**1. Policy/Actor Loss** (keep policy realistic):
```python
ratio = exp(log_prob_new - log_prob_old)
unrestricted_loss = ratio * advantage
clipped_loss = clip(ratio, 1-ε, 1+ε) * advantage
L_actor = -mean(min(unrestricted_loss, clipped_loss))

# Purpose: Move policy toward good actions,
#          but don't change too much (clip prevents wild swings)
```

**2. Value/Critic Loss** (accurate value estimates):
```python
value_target = reward + γ * V(s_{t+1})
L_value = mean_squared_error(V(s_t), value_target)

# Purpose: Make value network predict returns accurately
```

**3. Entropy Loss** (maintain exploration):
```python
entropy = -sum(π(a|s) * log(π(a|s)))
L_entropy = -entropy_weight * entropy

# Purpose: Prevent policy from becoming deterministic too early
```

**4. Combined Loss**:
```python
L_total = L_actor + 0.5 * L_value + 0.01 * L_entropy
```

### Optimization Step

```python
# For each minibatch of 32 transitions:
for epoch in range(K):  # K=3 typical
    minibatch = sample(buffer, size=32)
    
    loss = compute_total_loss(minibatch)
    
    # Backward pass
    gradient = backward(loss)
    
    # Update parameters
    θ_policy -= learning_rate * ∇L_actor
    θ_value -= learning_rate * ∇L_value
```

---

## Stage 7: Training Loop & Episodes

### Episode Structure

```python
for episode in range(200):  # 200 episodes
    circuit = sample_random_circuit()  # Pick random from 16 EPFL circuits
    state = get_circuit_features(circuit)
    
    for step in range(25):  # Max 25 steps per episode
        # Forward pass through networks
        action_probs = policy_network(state)
        value = value_network(state)
        
        # Sample action
        action = sample(action_probs)  # Stochastic selection
        
        # Execute action
        next_state, reward, done, info = env.step(action)
        
        # Store transition
        buffer.add({
            'observation': state,
            'action': action,
            'reward': reward,
            'next_observation': next_state,
            'done': done,
            'log_probability': log(action_probs[action]),
            'value_estimate': value
        })
        
        state = next_state
        
        if done:
            break  # Exit step loop
    
    # PPO update every K episodes
    if episode % update_frequency == 0:
        while buffer.has_data():
            # Sample minibatch
            minibatch = buffer.sample()
            
            # Compute losses
            losses = compute_losses(minibatch)
            
            # Update networks
            optimize(losses)
        
        buffer.clear()
```

### Training Hyperparameters

```
Learning settings:
├─ Optimizer: Adam
├─ Learning rate: 3e-4
├─ Batch size: 32
├─ PPO epochs: 3
├─ Clip ratio (ε): 0.2
├─ Value loss coeff: 0.5
├─ Entropy coeff: 0.01
├─ Discount (γ): 0.99
├─ GAE λ: 0.95
└─ Total episodes: 200

Circuit settings:
├─ Circuits per episode: 1 (random)
├─ Max steps/episode: 25
├─ Total circuits available: 16 (EPFL suite)
└─ Repeat circuits: Yes (random sampling with replacement)

Network sizes:
├─ GCN hidden: 64D → 128D
├─ Shared latent: 128D
├─ Policy head: 128→64→32→12
├─ Value head: 128→64→32→1
└─ Fused state: 256D
```

---

## Stage 8: Inference & Evaluation

### Setup
- **Model**: Trained policy & value networks
- **Circuits**: 16 test circuits from EPFL
- **Action selection**: Greedy (no sampling, deterministic)

### Inference Loop

```python
results = []

for circuit in test_circuits:
    state = get_circuit_features(circuit)
    
    for step in range(max_steps):  # e.g., 50 max steps
        # Forward through policy (no gradient)
        action_probs = policy_network(state)
        
        # Greedy action (take best, no randomness)
        action = argmax(action_probs)
        
        # Execute
        next_state, reward, done, info = env.step(action)
        
        state = next_state
        
        if reward > threshold or done:
            break
    
    # Measure final metrics
    final_luts = info['lut_count']
    final_depth = info['logic_depth']
    
    results.append({
        'circuit': circuit_name,
        'initial_luts': baseline_luts,
        'final_luts': final_luts,
        'reduction_percent': (baseline_luts - final_luts) / baseline_luts * 100,
        'depth': final_depth
    })
```

### Results Comparison

**Against Baselines**:
```
Benchmark Results:
┌─────────────┬──────────┬──────────┬──────────┐
│ Circuit     │ Baseline │ HybridSYN│ Improve% │
├─────────────┼──────────┼──────────┼──────────┤
│ adder       │   254    │   240    │  5.5%    │
│ multiplier  │  5913    │  5510    │  6.8%    │
│ arbiter     │  2722    │  2540    │  6.7%    │
│ ... (16 tot)│   ...    │   ...    │ ...      │
├─────────────┼──────────┼──────────┼──────────┤
│ Average     │   -      │   -      │  9.7%    │
└─────────────┴──────────┴──────────┴──────────┘

Comparison with other methods:
- vs RL-A2C: 9.7% better
- vs RL-PPO (AIG): 9.7% better  
- vs resyn2: 17.1% better
```

---

## Data Format Reference

### Circuit State (256D)

```
[128D AIG features]
  └─ From GCN processing of AND-inverter graph
     ├─ Node type embeddings
     ├─ Connectivity patterns
     └─ Global pooled representation

[128D MIG features]
  └─ From GCN processing of majority-inverter graph
     ├─ Majority gate embeddings
     ├─ Gate type distribution
     └─ Global pooled representation
```

### Action Space (12 actions)

```
Action 0-5: ABC operators
├─ 0: rewrite -l
├─ 1: refactor -l
├─ 2: balance
├─ 3: resub -l
├─ 4: ps
└─ 5: custom

Action 6-11: MIG operators
├─ 6: refactor
├─ 7: balance
├─ 8: resub
├─ 9: optimize
├─ 10: zero_cost
└─ 11: custom
```

### Reward Signal (scalar)

```
r_t = -(α × LUT_count + β × |logic_depth|)

Typical range:
  r_t ≈ -250 to -1000  (negative because optimization minimizes cost)
  
Good reward: larger (less negative) = fewer LUTs
Interpretation: r=−240 is better than r=−300
```

### Transition Tuple (8 elements)

```python
(
    state_t,            # [256D] - current circuit features
    action_t,           # scalar - selected action 0-11
    reward_t,           # scalar - performance metric
    state_{t+1},        # [256D] - circuit after action
    done,               # bool - episode termination
    log_probability,    # scalar - probability of action
    value_estimate,     # scalar - estimated return
    advantage           # scalar - reward - baseline
)
```

---

## Summary Table: Data Transformations

| Source | Component | Process | Destination | Shape |
|--------|-----------|---------|-------------|-------|
| AIGER file | Parser | Parse + split | AIG/MIG graphs | Variable |
| AIG graph | GCN | Embed nodes | Pooled features | 128D |
| MIG graph | GCN | Embed nodes | Pooled features | 128D |
| Features | Attention | Weight & fuse | Fused features | 256D |
| Features | Dense | Project | Shared latent | 128D |
| Latent | Policy head | Dense network | Action probs | 12D |
| Latent | Value head | Dense network | State value | 1D |
| Action + circuit | Environment | Execute operator | New circuit | Variable |
| Circuit | Evaluator | Map + measure | Metrics | (int, int) |
| Metrics | Reward fn | LUT + depth | Reward scalar | 1D |
| Transition | Buffer | Aggregate | Minibatch | (32, 8) |
| Minibatch | Loss fn | Compute loss | Gradients | - |
| Gradients | Optimizer | Update params | New weights | - |

