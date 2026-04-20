# Graph Convolutional Networks (GCN) - Complete Guide

## 1. What is a GCN?

### Simple Definition
**GCN** = Neural network layer that works on **graph data** (not just vectors/images)

Analogies:
- **CNN** (Convolutional Neural Network): Works on grid data (images)
  - Each pixel has neighbors in 2D grid
  - Convolution aggregates neighbor information
  
- **GCN** (Graph Convolutional Network): Works on graph data
  - Each node has neighbors in a graph
  - Message passing aggregates neighbor information

### Key Difference from Regular Neural Networks

**Regular Dense Layer**:
```
Input: vector [d_in]
Weight matrix: [d_in × d_out]
Output: vector [d_out]

Operation: output = input @ weight_matrix
No awareness of graph structure!
```

**GCN Layer**:
```
Input: node embeddings [N × d_in] + graph adjacency
For each node:
  • Aggregate messages from neighbors
  • Mix with own information
  • Apply nonlinearity
Output: updated embeddings [N × d_out]

Awareness of graph structure!
```

---

## 2. What Does GCN Do?

### The Core Operation: Message Passing

**Step-by-step for ONE node**:

```
Node v has neighbors: u1, u2, u3

Step 1: Get neighbor embeddings
  emb(u1) = [0.1, 0.2, ..., 0.5]  [d_in dimensions]
  emb(u2) = [0.3, 0.1, ..., 0.2]
  emb(u3) = [0.2, 0.4, ..., 0.1]

Step 2: Aggregate from neighbors
  neighbor_message = AGGREGATE(emb(u1), emb(u2), emb(u3))
  
  Common aggregations:
  ├─ MEAN:   (emb(u1) + emb(u2) + emb(u3)) / 3
  ├─ MAX:    element-wise max(emb(u1), emb(u2), emb(u3))
  └─ SUM:    emb(u1) + emb(u2) + emb(u3)

Step 3: Combine with own information
  combined = neighbor_message + emb(v)  # Add own embedding

Step 4: Apply transformation
  new_emb(v) = ReLU(combined @ Weight_matrix + bias)
  
  Result: new embedding for node v
```

### Intuition

**For circuit graphs**:
```
Gate A is an AND gate
Its neighbors: 
  - Input signal 1 → AND inputs
  - Input signal 2 → AND inputs
  - Output gate B → AND output goes here

GCN learns:
  "Gate A's behavior depends on its neighbors"
  Aggregates neighbor patterns
  → Creates better representation of Gate A
```

---

## 3. Why Do We Need GCN?

### Problem: Circuits are Graphs, Not Vectors

**Can't use regular neural network directly**:
```
Regular NN: takes fixed-size vector input
Circuit AIG: 254 AND gates
Circuit MIG: 300 majority gates
Different sizes! How do we feed to a fixed NN?

Solution 1 (Bad): Pad to max size
├─ Wastes memory
├─ Loses information about actual size
└─ Inefficient

Solution 2 (Worse): Flatten entire adjacency matrix
├─ 254×254 = 64K dimensional input
├─ Explodes with larger circuits
├─ No generalization to different sizes
└─ Computationally expensive

Solution 3 (Good): Use GCN!
├─ Works on any graph size
├─ Learns local patterns
├─ Efficient message passing
└─ Generalizes across circuit sizes
```

### Advantages of GCN

1. **Variable Graph Size**
   - Input: N nodes (any N)
   - Process with GCN
   - Pool to fixed output: [d_out]
   - Works for any graph!

2. **Locality Awareness**
   - Learns based on local graph structure
   - "This gate's behavior depends on neighbors"
   - Not just global features

3. **Parameter Efficiency**
   - Same weights for all nodes
   - Scales to large graphs
   - Weight sharing!

4. **Hierarchical Learning**
   - Layer 1: Learn 1-hop neighborhoods
   - Layer 2: Learn 2-hop neighborhoods
   - Composite patterns

---

## 4. How GCN Works (Mathematical Detail)

### Single GCN Layer

**Notation**:
```
X ∈ ℝ^(N×d_in)      Graph with N nodes, each d_in dimensions
A ∈ ℝ^(N×N)         Adjacency matrix (1 if nodes connected, 0 otherwise)
W ∈ ℝ^(d_in×d_out)  Weight matrix to learn
```

**Forward pass**:
```
Normalized adjacency: Ã = D^(-1/2) A D^(-1/2)
  (where D = degree matrix, added for numerical stability)

Message aggregation: H = Ã @ X @ W
  Step by step:
  1. X @ W: Transform each node's features [N × d_out]
  2. Ã @ (X @ W): Aggregate neighbor transformed features

Add bias and activation: Y = ReLU(H + b)
```

**What this means**:
```
For each node i:
  aggregated = mean_of_neighbors(transformed_neighbor_features)
  new_embedding[i] = ReLU(aggregated + bias)
```

### Two-Layer GCN (Like HybridSYN Uses)

```
Input: X [N × d_in]  (circuit graph, N=number of gates, d_in=input features)

Layer 1:
  H1 = ReLU(Ã @ X @ W1 + b1)  [N × d_hidden]
  Output: N nodes with d_hidden embeddings
  
Layer 2:
  H2 = ReLU(Ã @ H1 @ W2 + b2)  [N × d_output]
  Output: N nodes with d_output embeddings (e.g., 128D)

Pooling:
  global_avg = mean(H2)  [d_output]
  global_max = max(H2)   [d_output]
  circuit_embedding = concat(global_avg, global_max)  [2*d_output]
```

---

## 5. How GCN Applies to HybridSYN Specifically

### The Circuit Graph

**AIG Representation**:
```
Nodes:
  ├─ AND gates (typical nodes)
  ├─ NOT gates (inverters)  
  └─ Primary inputs/outputs

Edges:
  └─ Signal connections between gates

Example (4-bit adder):
     IN0 ──┐
            AND ──┐
     IN1 ──┤     ├─ AND ──┐
                          ├─ OR ──→ OUT0
     IN2 ──┐     ┌─ AND ──┤
            AND ──┤
     IN3 ──┤
            └─────────────→ OUT1
```

### GCN Processing Pipeline for HybridSYN

```
Step 1: Create Graph Structure
─────────────────────────────────
Circuit representation:
  Nodes: gates {AND1, AND2, ..., AND254}
  Edges: connections between gates
  
Encode as:
  X = [node_features]  [254 × d_in]
    Where each row has features like:
    └─ gate_type (AND=1, OR=2, NOT=3, etc.)
    
  A = [adjacency_matrix]  [254 × 254]
    Where A[i,j] = 1 if gate_i connects to gate_j


Step 2: GCN Layer 1
──────────────────
Input: X [254 × d_in]
Process: H1 = ReLU(Ã @ X @ W1)
Output: H1 [254 × d_hidden]

What happened:
  ├─ For each gate: received messages from neighboring gates
  ├─ Combined with own features
  ├─ Applied ReLU nonlinearity
  └─ Now each gate has rich representation of its local context


Step 3: GCN Layer 2
──────────────────
Input: H1 [254 × d_hidden]
Process: H2 = ReLU(Ã @ H1 @ W2) + BatchNorm
Output: H2 [254 × 128D]

What happened:
  ├─ Higher-level pattern recognition
  ├─2-hop neighborhoods aggregated
  ├─ Each gate now aware of its neighbors' neighbors
  └─ Captures more complex structural patterns


Step 4: Pooling
──────────────
Input: H2 [254 × 128D]

Global Average Pooling:
  avg_pool = mean(H2)  [128D]
  Interpretation: "What's the average token across all gates?"

Global Max Pooling:
  max_pool = max(H2)   [128D]
  Interpretation: "What's the most salient feature?"

Concatenate:
  circuit_features = concat(avg_pool, max_pool)  [256D]

Final output: Single 256D vector representing entire circuit!
```

### Why This Works for Circuits

**GCN learns circuit patterns**:
```
Example patterns it might learn:

Pattern 1: Critical Path Detection
├─ Nodes in long chains aggregate more information
├─ Lead to different embedding
└─ Network learns "deep gates are different"

Pattern 2: Fanout Concentration
├─ Gate driving many outputs gets many neighbor messages
├─ Rich embedding representation
└─ Network learns "critical gates are different"

Pattern 3: Gate Type Locality
├─ AND gates near AND gates act differently
├─ Than AND gates near OR gates
└─ Network learns local patterns

Pattern 4: Structural Motifs
├─ Common patterns (adders, multipliers)
├─ Network recognizes recurring structures
└─ Generalizes across circuits
```

---

## 6. Advantages & Disadvantages

### ✅ Advantages

1. **Variable Size Handling**
   - Works on 100-node and 10,000-node graphs
   - No need for padding/resizing
   - Elegant solution

2. **Structural Awareness**
   - Captures connectivity patterns
   - Learns graph-aware features
   - Not just node-level statistics

3. **Generalization**
   - Same weights for all nodes
   - Scales to unseen circuit sizes
   - Transfer learning possible

4. **Interpretability**
   - Can visualize attention patterns
   - Understand which connections matter
   - Explainable decisions

5. **Computational Efficiency**
   - Sparse matrix multiplication (if using sparse graphs)
   - Message passing is parallelizable
   - Practical for large circuits

### ❌ Disadvantages

1. **Over-smoothing**
   - Too many layers → all nodes become similar
   - Usually limited to 2-3 layers
   - (HybridSYN uses 2 layers)

2. **Spectral Properties**
   - Requires normalized adjacency matrix
   - Some graph properties not captured
   - Bilateral connection assumption

3. **Scalability Limits**
   - Very large graphs (millions of nodes)
   - May need approximations
   - Memory constraints

4. **Hyperparameter Sensitivity**
   - Hidden dimension size matters
   - Pooling strategy affects output
   - Requires tuning

---

## 7. Practical Example: Small Circuit

### Concrete Example: 2-bit Multiplier

**Circuit**:
```
Inputs: A0, A1, B0, B1
Operation: result = (A0*B0) + (A1*B1)
Simplified to 3 AND gates, 1 OR gate

Nodes: {AND1, AND2, OR3}
Edges: A0→AND1, B0→AND1, A1→AND2, B1→AND2, 
       AND1→OR3, AND2→OR3

Adjacency Matrix:
     AND1  AND2  OR3
AND1 [  0     0    1  ]
AND2 [  0     0    1  ]
OR3  [  0     0    0  ]
```

**Node Features (Input)**:
```
AND1: [gate_type=1, fanin=2, fanout=1]  = [1, 2, 1]
AND2: [gate_type=1, fanin=2, fanout=1]  = [1, 2, 1]
OR3:  [gate_type=2, fanin=2, fanout=0]  = [2, 2, 0]

X = [[1, 2, 1],
     [1, 2, 1],
     [2, 2, 0]]  [3 × 3]
```

**GCN Layer 1**:
```
W1 = random weight matrix [3 × 4]

Transform features: X @ W1  [3 × 4]
Aggregate from neighbors: Ã @ (X @ W1)  [3 × 4]

Result: each gate has 4D embedding
├─ AND1 gets its features + information from neighbors
├─ AND2 gets its features + information from neighbors  
└─ OR3 gets aggregated signals from AND1 and AND2
```

**GCN Layer 2**:
```
W2 = random weight matrix [4 × 128]

Same process: Ã @ H1 @ W2  [3 × 128]

Result: each gate has 128D embedding
├─ AND1: 128D representation
├─ AND2: 128D representation
└─ OR3:  128D representation (rich info from inputs)
```

**Pooling**:
```
avg_pool = mean(H2) = (AND1 + AND2 + OR3) / 3  [128D]
max_pool = max(H2)  = element-wise max         [128D]

circuit_feature = concat(avg_pool, max_pool)  [256D]

Output: Single 256D vector representing 2-bit multiplier!
```

---

## 8. Comparison: With vs Without GCN

### Without GCN (Naive Approach)

```python
# Problem: circuits have different sizes
circuit1 = [254 AND gates]     # AIG representation
circuit2 = [300 MAJ gates]     # MIG representation

# Solution: Flatten and pad
X1_flat = pad_to_max(flatten(circuit1))  [10000D]  # Wasted space!
X2_flat = pad_to_max(flatten(circuit2))  [10000D]  # Wasted space!

FC_layer: [10000] → [128]  # 1.28M parameters!
Network is huge, inefficient, doesn't generalize
```

### With GCN (HybridSYN Approach)

```python
# Same circuits, different handling
circuit1 = [254 nodes, edges]   # AIG graph
circuit2 = [300 nodes, edges]   # MIG graph

GCN_layer1: [N × d_in] → [N × d_hidden]  # Any N!
GCN_layer2: [N × d_hidden] → [N × 128]
Pooling:    [N × 128] → [128]            # Fixed output!

Output: Both produce [128D] vector
Network is small, efficient, generalizes!
```

---

## 9. Training GCN

### What Gets Learned?

**Weight matrices**:
```
W1: [d_in × d_hidden]         Learned during training
W2: [d_hidden × d_output]     Learned during training
b1, b2: bias vectors           Learned during training
```

**What they learn**:
```
W1 learns: "How to combine node features and neighbor info"
W2 learns: "How to refine and abstract patterns"
Biases:    "Channel-wise adjustments"

After training:
├─ Critical gates have distinct embeddings
├─ Pattern-rich regions have similar embeddings
├─ Structural features are captured
└─ Network can predict actions based on these features
```

### Training Process

```
For each training step:
  1. Forward pass: circuit → GCN → [128D] features
  2. Use features for task (action selection, value prediction)
  3. Compute loss based on task (PPO loss in HybridSYN)
  4. Backward pass: compute ∇L with respect to weights
  5. Update W1, W2, b1, b2 via SGD
  
After 200 episodes of training:
├─ GCN learns good circuit representations
├─ Can distinguish circuit types
├─ Captures optimization-relevant patterns
└─ Policy network makes better decisions
```

---

## 10. GCN in HybridSYN Summary

### The Full Pipeline

```
AIGER Circuit File
     ↓
Parse → AIG Graph (254 nodes)
     ↓
GCN Layer 1: Aggregate from neighbors
  [254 × input_features] → [254 × 64]
     ↓
GCN Layer 2: Refine patterns
  [254 × 64] → [254 × 128]
     ↓
Pooling: Compress to circuit level
  [254 × 128] → [128]
     ↓
Circuit AIG Feature Vector [128D]

SAME PROCESS FOR MIG:
Parse → MIG Graph (300 nodes)
  [GCN] → [128D]

FINAL OUTPUT:
[128D AIG features] + [128D MIG features] = [256D state]
This goes to policy network for decision!
```

### Why GCN Specifically for This Task

1. **Circuits are naturally graphs** - perfect match
2. **Variable sizes** - GCN handles this
3. **Local patterns matter** - message passing captures this
4. **Need fixed output** - pooling provides this
5. **Efficient** - scales well
6. **State-of-the-art** - proven effective in many domains

---

## 11. Key Takeaways

| Question | Answer |
|----------|--------|
| **What is GCN?** | Neural network layer operating on graphs via message passing |
| **What does it do?** | Transforms node features by aggregating neighbor information |
| **Why need it?** | Circuits are graphs; GCN solves variable size + structure awareness |
| **How apply?** | For each gate: get neighbors' embeddings → aggregate → transform |
| **Output?** | Fixed-size circuit representation regardless of input size |
| **Advantage?** | Generalizes to unseen circuit sizes; structure-aware; efficient |
| **In HybridSYN?** | Extract [128D] features from AIG and MIG independently |

---

## 12. Visual Summary

```
Regular Neural Network              Graph Convolutional Network
─────────────────────              ──────────────────────────

Input: Fixed vector [d_in]         Input: Variable graph N nodes
   ↓                                  ↓
Fixed architecture                 Node aggregation (message pass)
   ↓                                  ↓
Hidden layers                       GCN layer 1, 2
   ↓                                  ↓
Output: [d_out]                    Node embeddings [N × d_out]
                                      ↓
                                    Pooling (fixed output)
                                      ↓
                                    Output: [d_out]

Limitation:                         Advantages:
├─ Fixed size only                 ├─ Variable size
├─ No graph awareness              ├─ Graph-aware
└─ Doesn't generalize              ├─ Generalizes
                                   └─ Efficient
```

