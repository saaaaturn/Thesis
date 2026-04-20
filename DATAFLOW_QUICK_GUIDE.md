# HybridSYN Data Flow - Quick Checklist & Navigation Guide

## 📚 Documentation Created

### 1. Visual Diagrams (4 Mermaid Diagrams)
- ✅ **Complete Data Flow**: AIGER → Optimization → Results
- ✅ **Feature Extraction & Network Architecture**: GCN + Cross-Attention + Policy/Value heads
- ✅ **Training Loop**: Episode structure and PPO update cycle
- ✅ **Data Shapes & Dimensions**: Exact tensor sizes through pipeline

### 2. Detailed Documents
- ✅ **DATAFLOW.md**: ASCII diagrams + 7-stage detailed breakdown
- ✅ **DATAFLOW_DETAILED.md**: In-depth explanation of each stage with code examples
- ✅ **Session memory files**: Quick reference guides

---

## 🔍 Understanding the System

### Key Concept: **256D Circuit Representation**

Every circuit in HybridSYN is converted to a 256-dimensional vector:

```
Circuit Feature Vector [256D]
├─ AIG GCN embedding [128D]    (What AND-gates look like)
├─ MIG GCN embedding [128D]    (What MAJORITY-gates look like)
```

This 256D vector is the "language" the neural networks understand.

---

### The 7-Stage Pipeline

| # | Stage | Input | Output | Key Insight |
|---|-------|-------|--------|-------------|
| 1 | **Load** | AIGER file | AIG/MIG graphs | Two representations |
| 2 | **Extract** | Graphs | AIG [128D] + MIG [128D] | GCN learns patterns |
| 3 | **Fuse** | AIG + MIG | Combined [256D] | Attention learns weightings |
| 4 | **Decide** | State [256D] | Actions [12D] | RL agent picks operator |
| 5 | **Execute** | Action + circuit | New circuit | Operator transforms circuit |
| 6 | **Learn** | Transitions | Updated weights | PPO trains networks |
| 7 | **Train** | Episodes | Optimized policy | Repeat 200x |

---

## 🎯 Critical Data Formats

### Circuit State (256D vector)
```
What it represents: Structural features of the circuit
├─ 128D: How the circuit looks as AND gates (AIG view)
├─ 128D: How the circuit looks as MAJORITY gates (MIG view)
```

### Action Space (12 choices)
```
What it means: Which operator to apply
├─ 0-5: ABC synthesis operators (for AIG)
├─ 6-11: MIG synthesis operators
```

### Reward Signal (scalar)
```
What it means: How good was that action?
Formula: r_t = -(α × LUT_count + β × |logic_depth|)
Negative value = better result (fewer LUTs)
```

### Episode Transition (stored in buffer)
```
What it records: (state, action, reward, next_state, done, ...)
Purpose: Used for PPO learning updates
```

---

## 🧠 Neural Network Architecture

```
Input: Circuit state [256D]
   ↓
Shared FC layer: 256D → 128D
   ├─ Policy Head: 128→64→32→12  (softmax → action probabilities)
   └─ Value Head: 128→64→32→1    (scalar → state value estimate)
   
Output: 
   ├─ Action probabilities [12D]
   └─ Value estimate [1D]
```

---

## 🔄 Training Loop Structure

```
For 200 episodes:
  1. Pick random circuit
  2. For up to 25 steps:
     - Observe circuit state
     - Sample action (stochastic)
     - Apply operator
     - Store experience
  
  When buffer fills:
  3. PPO Update:
     - Sample minibatches
     - Compute losses
     - Update networks
     - Clear buffer
     - Repeat
```

---

## 📊 What Gets Measured

**During Training**:
- Policy network learns to pick good operators
- Value network learns to estimate circuit quality
- Replay buffer stores experiences

**During Evaluation**:
- Test on 16 benchmark circuits
- Use trained policy (greedy, no randomness)
- Measure LUT reduction %
- Compare to baselines

**Expected Results**:
- 9-17% improvement in LUT count
- Faster convergence than single-representation approaches
- Better generalization across diverse circuit types

---

## 🔗 Data Flow Sequences

### Sequence 1: One Environment Step
```
1. Extract features from current circuit → state [256D]
2. Forward through policy network → action probabilities [12D]
3. Sample action → integer 0-11
4. Apply action to circuit → modified circuit
5. Evaluate new circuit → LUT count, depth
6. Calculate reward → scalar value
7. Create transition tuple → store in buffer
```

### Sequence 2: PPO Update
```
1. Sample minibatch from buffer → 32 transitions
2. Compute GAE advantages → 32 advantage values
3. Compute policy loss → gradient for policy network
4. Compute value loss → gradient for value network
5. Compute entropy bonus → encourage exploration
6. Total loss = policy loss + value loss + entropy
7. Backward pass → compute gradients
8. Optimizer step → update network weights
9. Repeat for K epochs
```

### Sequence 3: Full Episode
```
1. Select random circuit
2. Extract initial state
3. For 25 steps:
   - Policy decision
   - Environment execution
   - Transition storage
4. If buffer full → PPO update
5. Clear buffer
6. Next episode
```

---

## 💡 Key Insights

### Why Two Representations?
- **AIG**: Compact, good for area optimization
- **MIG**: Better for arithmetic circuits, good depth control
- **Hybrid**: Learn which to use for each circuit

### Why Cross-Attention?
- Learns circuit-dependent weighting
- Arithmetic circuits → favor MIG (e.g., 70% MIG, 30% AIG)
- Control circuits → favor AIG (e.g., 70% AIG, 30% MIG)

### Why PPO?
- Stable policy updates (clipping prevents wild swings)
- Off-policy ability (uses replay buffer efficiently)
- Actor-Critic structure (policy + value networks)

### Why GCN for Features?
- Captures local graph structure
- Aggregates neighbor information
- Learns hierarchical representations
- Works on graphs of variable size

---

## 📝 Usage Guide for Each Document

### If you need to understand...

**The big picture**: 
→ Read this file (quick checklist)

**How data moves through system**:
→ Read DATAFLOW.md (diagrams + stage breakdown)

**Implementation details**:
→ Read DATAFLOW_DETAILED.md (code examples + math)

**Visual architecture**:
→ Look at the 4 Mermaid diagrams

**Quick reference**:
→ Check session memory files (data_flow_summary.md)

---

## 🚀 Next Steps for Implementation

Once you understand the data flow, you'll need to implement:

1. **interface.cpp** - C++/Python bridge
   - Export circuit graphs as Python data
   - Execute ABC/MIG operators
   - Evaluate metrics (LUT, depth)

2. **HybridSYN Gymnasium Environment**
   - Implement `reset()`, `step()`, `render()`
   - State: 256D circuit features
   - Actions: 12 operators
   - Rewards: LUT-based scoring

3. **GCN Feature Extractor** (Python/PyTorch)
   - Graph convolution layers
   - Pooling operations
   - Statistical features extraction

4. **RL Training Script**
   - PPO integration with stable-baselines3
   - Hyperparameter tuning
   - Training and validation

5. **Cross-Attention Module**
   - Query/Key/Value computations
   - Weighted feature fusion
   - Attention visualization

6. **Evaluation & Results**
   - Baseline comparisons
   - Metric logging
   - Visualization plots

---

## 📐 Dimensions Reference

```
Circuit representation:
├─ AIG node embeddings: [N × 128D]  where N = AND gate count
├─ MIG node embeddings: [K × 128D]  where K = MAJ gate count
├─ Pooled AIG: [128D]
├─ Pooled MIG: [128D]
├─ Statistics: [10D]
├─ After fusion: [256D]
├─ Policy output: [12D]
└─ Value output: [1D]

Batch sizes:
├─ Episode: 1 circuit at a time
├─ Steps/episode: 25 max
├─ Minibatch: 32 transitions
└─ Total training: 200 episodes × 16 circuits × 25 steps (max)
```

---

## ✅ Verification Checklist

When you implement each stage, verify:

- [ ] Stage 1: import.cpp creates valid AIG/MIG graphs
- [ ] Stage 2: GCN produces [128D, 128D] feature vectors
- [ ] Stage 3: Attention fuses to [256D]
- [ ] Stage 4: Policy outputs [12D], Value outputs [1D]
- [ ] Stage 5: Operators change circuit, metrics vary
- [ ] Stage 6: Reward reflects LUT reduction
- [ ] Stage 7: Training loop runs 200 episodes
- [ ] Stage 8: Inference compares to baselines

---

## 📞 Quick Reference Summary

**What is HybridSYN?**
- RL framework for circuit optimization
- Uses PPO algorithm
- Switches between AIG and MIG representations
- Achieves 9-17% better results than baselines

**Key data structure?**
- 256D feature vector (128D AIG + 128D MIG)

**What does network do?**
- Policy: picks 1 of 12 operators
- Value: estimates circuit quality

**What does environment do?**
- Applies operator
- Measures LUT count & depth
- Returns reward signal

**How does it learn?**
- Collects transitions
- Stores in replay buffer
- PPO updates when buffer full
- Repeats for 200 episodes

