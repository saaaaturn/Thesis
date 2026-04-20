# Stage 2: Feature Extraction from AIG/MIG - Implementation Guide

## Overview: What You're Doing Next

After loading circuits into AIG/MIG (Stage 1 ✅), you now extract:

```
Stage 2: Feature Extraction
├─ From AIG: Node features + Adjacency matrix
├─ From MIG: Node features + Adjacency matrix
└─ From both: Statistical characteristics
```

This prepares **raw graph data** for GCN processing in the next stage.

---

## 1. Understanding What You Need to Extract

### Node Features (What Mockturtle Gives You)

For **EACH gate/node** in the circuit, extract:

```
Node features = [
  gate_type,      // 0=AND, 1=OR, 2=NOT, 3=MAJ, etc.
  fanin_count,    // How many inputs feed into this gate
  fanout_count,   // How many gates this input feeds
  is_critical,    // Is this on critical path (optional)
  logic_depth,    // Distance from primary inputs
]
```

Example:
```
Gate 0 (AND): [type=0, fanin=2, fanout=3, depth=1]
Gate 1 (AND): [type=0, fanin=2, fanout=1, depth=2]
Gate 2 (OR):  [type=1, fanin=3, fanout=1, depth=3]
Gate 3 (NOT): [type=2, fanin=1, fanout=2, depth=4]
...
```

### Adjacency Matrix (Graph Connectivity)

**What it represents**: Which nodes (gates) connect to which

```
Adjacency matrix A [N × N]:
  A[i][j] = 1 if gate_i connects to gate_j
  A[i][j] = 0 otherwise

Example for 3 gates:
      Gate0  Gate1  Gate2
Gate0 [  0      0      1  ]   Gate0 connects to Gate2
Gate1 [  0      0      1  ]   Gate1 connects to Gate2
Gate2 [  0      0      0  ]   Gate2 connects to nothing
```

### Statistical Characteristics

Global circuit properties (extracted once, shared):

```
Statistics = [
  total_gates,          // N (number of nodes)
  total_edges,          // M (number of connections)
  circuit_depth,        // Max logic depth
  avg_fanin,            // Average inputs per gate
  avg_fanout,           // Average outputs per gate
  gate_type_histogram,  // % of each gate type
  is_arithmetic,        // Heuristic: detect circuit type
  // ... other metrics
]
```

---

## 2. What Format to Use

### Node Features Matrix [N × F] Format

```python
# For circuit with N gates and F features per gate
node_features = numpy array or PyTorch tensor [N × F]

Example:
N = 254 (total AND gates in 256-bit adder)
F = 5 (features: type, fanin, fanout, critical, depth)

node_features shape: [254 × 5]
node_features[0] = [0, 2, 3, 0, 1]  # Gate 0: AND, 2 inputs, 3 outputs, not critical, depth 1
node_features[1] = [0, 2, 1, 1, 2]  # Gate 1: AND, 2 inputs, 1 output, critical, depth 2
...
```

### Adjacency Matrix [N × N] Format

**Two storage options:**

**Option 1: Dense Matrix** (Simple, uses more RAM)
```python
adjacency = numpy array [N × N]
adjacency[i][j] = 1 if edge exists, 0 otherwise

For 254 gates: 254 × 254 = 64,516 elements
Memory: ~256 KB (if 32-bit integers)
```

**Option 2: Sparse Matrix** (Efficient, smaller RAM)
```python
# COO (Coordinate) format: only store non-zero entries
adjacency_coo = {
  'row': [0, 0, 1, 1, 2, ...],
  'col': [2, 5, 2, 7, 3, ...],
  'data': [1, 1, 1, 1, 1, ...]
}

For 254 gates with ~500 edges:
Memory: ~500 elements only
```

**Recommendation for HybridSYN**: Use COO sparse format (more efficient)

### Statistical Features Vector [S] 

```python
statistics = numpy array [S] where S = number of statistics

Example (S = 15):
statistics = [
  254,        # total_gates (N)
  500,        # total_edges (M)
  51,         # max_depth
  2.0,        # avg_fanin
  2.5,        # avg_fanout
  0.95,       # % AND gates
  0.02,       # % OR gates
  0.03,       # % NOT gates
  0.0,        # % MAJ gates
  100,        # critical path length (signals)
  256,        # num inputs
  129,        # num outputs
  1.0,        # arithmetic score (heuristic)
  0.0,        # control score (heuristic)
  0.5,        # gate density
]
```

---

## 3. How to Extract from Mockturtle

### Getting Node Count & Gate Information

```cpp
// From AIG
aig_network aig;
// ... aig is populated after read_aiger ...

// Access nodes
aig.foreach_node([&](auto node) {
  // node is a handle to a gate/node
  
  uint32_t fanin = aig.fanin_size(node);    // Number of inputs
  
  // Get fanout count - requires traversing
  // (Mockturtle: fanout not directly available, you track it)
  
  // Determine gate type
  if (aig.is_and(node)) { /* type = AND */ }
  if (aig.is_pi(node)) { /* type = PI (primary input) */ }
  if (aig.is_po(node)) { /* type = PO (primary output) */ }
});

// Total gate count
uint32_t num_gates = aig.num_gates();
```

### Getting Adjacency Information

```cpp
// Build adjacency by traversing connections
std::map<std::pair<int,int>, int> adjacency_map;

aig.foreach_node([&](auto target_node) {
  // For each input to this node
  aig.foreach_fanin(target_node, [&](auto fanin_signal) {
    // Get the source node from the signal
    auto source_node = fanin_signal.index;
    
    // Record edge: source → target
    adjacency_map[{source_node, target_node}] = 1;
  });
});

// Convert to matrix format:
// adjacency_map now has all edges
// Can be converted to COO sparse format
```

### Computing Statistics

```cpp
// Statistics to compute:

int total_gates = aig.num_gates();
int total_pis = aig.num_pis();      // Primary inputs
int total_pos = aig.num_pos();      // Primary outputs
int total_signals = aig.num_signals(); // All signals

// Compute fanin/fanout statistics
int total_fanin = 0;
int gate_count = 0;
aig.foreach_node([&](auto node) {
  if (aig.is_and(node)) {
    total_fanin += aig.fanin_size(node);
    gate_count++;
  }
});
double avg_fanin = (double)total_fanin / gate_count;

// Compute depth using levelization
std::map<int, int> node_level;
aig.foreach_node([&](auto node) {
  int max_fanin_level = 0;
  aig.foreach_fanin(node, [&](auto fanin_signal) {
    int fanin_node = fanin_signal.index;
    max_fanin_level = std::max(max_fanin_level, 
                               node_level[fanin_node]);
  });
  node_level[node] = max_fanin_level + 1;
});

int circuit_depth = *std::max_element(node_level | ranges::views::values);
```

---

## 4. Data Flow: From Mockturtle to Python

You have options for moving this data to Python:

### Option A: Save to Files (Cleaner Separation)

```cpp
// In interface.cpp, save to JSON or CSV

#include <nlohmann/json.hpp>

nlohmann::json circuit_data;
circuit_data["aig"] = {
  {"node_features", node_features_matrix},
  {"adjacency", adjacency_coo},
  {"statistics", statistics_vector}
};
circuit_data["mig"] = {
  {"node_features", mig_node_features},
  {"adjacency", mig_adjacency_coo},
  {"statistics", mig_statistics_vector}
};

std::ofstream out("circuit_features.json");
out << circuit_data.dump(2);
```

### Option B: Direct C++/Python Bridge (Via Pybind11)

```cpp
// Expose Mockturtle functions to Python directly
PYBIND11_MODULE(hybridsyn_cpp, m) {
  m.def("extract_aig_features", &extract_aig_features);
  m.def("extract_mig_features", &extract_mig_features);
}
```

### Option C: Simple Text Output (Debugging)

```cpp
// Write matrices to simple text files
// node_features.txt: one node per line, space-separated values
// adjacency.txt: source target (edge list format)
// statistics.txt: one stat per line
```

**Recommendation**: Start with **Option C** (text files) for debugging, then move to **Option A** (JSON) for production.

---

## 5. Pseudocode: Complete Feature Extraction

```cpp
// interface.cpp - Feature extraction from AIG/MIG

#include <vector>
#include <map>
#include <mockturtle/io/aiger_reader.hpp>
#include <mockturtle/networks/aig.hpp>
#include <mockturtle/networks/mig.hpp>

using namespace mockturtle;

struct CircuitFeatures {
  // Node features: each gate's properties
  std::vector<std::vector<float>> node_features;
  
  // Adjacency: sparse edge list format
  std::vector<std::pair<int, int>> edges;
  
  // Statistics: global circuit properties
  std::vector<float> statistics;
};

template<typename Network>
CircuitFeatures extract_features(Network& network) {
  CircuitFeatures features;
  
  // 1. Extract node features
  int num_nodes = network.num_gates();
  std::map<auto, int> node_to_index;
  int node_idx = 0;
  
  network.foreach_node([&](auto node) {
    if (network.is_and(node) || network.is_maj(node)) {
      // Get fanin
      int fanin = network.fanin_size(node);
      
      // Get fanout (need to track manually)
      int fanout = 0; // TODO: track during edge traversal
      
      // Gate type
      int gate_type = network.is_and(node) ? 0 : 3; // 0=AND, 3=MAJ
      
      // Store features
      features.node_features.push_back({
        (float)gate_type,
        (float)fanin,
        (float)fanout,
        0.0, // TODO: critical path
        0.0  // TODO: depth
      });
      
      node_to_index[node] = node_idx++;
    }
  });
  
  // 2. Extract edges (adjacency)
  network.foreach_node([&](auto target) {
    network.foreach_fanin(target, [&](auto fanin_signal) {
      auto source = fanin_signal.index;
      features.edges.push_back({
        node_to_index[source],
        node_to_index[target]
      });
    });
  });
  
  // 3. Compute statistics
  features.statistics = {
    (float)num_nodes,
    (float)features.edges.size(),
    (float)network.num_pis(),
    (float)network.num_pos(),
    // ... more stats ...
  };
  
  return features;
}

// Main function
int main() {
  aig_network aig;
  mig_network mig;
  
  // Load circuits (Stage 1)
  read_aiger("circuit.aig", aiger_reader(aig));
  read_aiger("circuit.aig", aiger_reader(mig));
  
  // Extract features (Stage 2)
  auto aig_features = extract_features(aig);
  auto mig_features = extract_features(mig);
  
  // Save to file (or use in Python)
  // save_features("aig_features.txt", aig_features);
  // save_features("mig_features.txt", mig_features);
  
  return 0;
}
```

---

## 6. What Each Extraction Gives You

### Node Features [N × F]

```
What it enables:
├─ GCN input: initial node embeddings
├─ Pattern recognition: gate type, connectivity patterns
├─ Interpretability: understand which features matter
└─ Training signal: learn from gate properties
```

### Adjacency Matrix [N × N] or Edge List

```
What it enables:
├─ Graph structure: defines message passing paths
├─ Connectivity awareness: gates know their neighbors
├─ Sparse operations: efficient computation
└─ Pattern propagation: local patterns aggregate
```

### Statistics Vector [S]

```
What it enables:
├─ Circuit-level features: global properties
├─ Attention weights: learn which representation matters
├─ Task difficulty: understand circuit complexity
└─ Heuristics: arithmetic vs control circuit classification
```

---

## 7. Verification Checklist

After extraction, verify:

- [ ] Node features matrix has shape [N × F] where N = num_gates
- [ ] Feature dimensions correct (5-10 features per node recommended)
- [ ] Adjacency has ~2-3 edges per gate on average (depends on circuit)
- [ ] Statistics vector computed (should be circuit-specific)
- [ ] Both AIG and MIG extracted independently
- [ ] No NaN or Inf values in features
- [ ] Edge list doesn't exceed square matrix size
- [ ] Feature ranges reasonable (normalized or unnormalized?)

---

## 8. Feature Normalization (Optional but Recommended)

Before sending to GCN, normalize features to [0, 1] or mean=0, std=1:

```python
# Option 1: Min-Max scaling [0, 1]
node_features_normalized = (node_features - node_features.min()) / \
                          (node_features.max() - node_features.min())

# Option 2: Standardization (mean=0, std=1)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
node_features_normalized = scaler.fit_transform(node_features)

# Option 3: No normalization (for depth, gate_type which are discrete)
# Leave as-is
```

---

## 9. Implementation Steps

### Step 1: Extend import.cpp (or create interface.cpp)
```cpp
// Add feature extraction functions
// Returns Node Features, Adjacency, Statistics
```

### Step 2: Define Output Format
```cpp
// Decide on file format (text, JSON, binary)
// Write extraction functions
```

### Step 3: Extract AIG Features
```cpp
// Call extract_features(aig)
// Save to files
```

### Step 4: Extract MIG Features
```cpp
// Call extract_features(mig)
// Save to files
```

### Step 5: Verify in Python
```python
# Load extracted features
# Check shapes, ranges, validity
# Visualize graph if needed
```

### Step 6: Ready for GCN
```python
# Use node_features and adjacency in GCN input
# Feed statistics to attention layer
```

---

## 10. Summary

**What you're extracting:**

| Component | From | Format | Size | Purpose |
|-----------|------|--------|------|---------|
| Node Features | AIG | Matrix [N×F] | 254×5 | GCN input |
| Adjacency | AIG | Sparse edges | ~500 pairs | Graph structure |
| Statistics | AIG | Vector [S] | 15-20 | Circuit properties |
| Node Features | MIG | Matrix [N×F] | varies×5 | GCN input |
| Adjacency | MIG | Sparse edges | ~M pairs | Graph structure |
| Statistics | MIG | Vector [S] | 15-20 | Circuit properties |

---

## Next Steps

Once you have these extracted:

**Stage 3**: Feed to GCN
- Node features → GCN Layer 1
- Adjacency → Message passing
- Output: [128D] embedding

Shall I show you how to implement the GCN layer next, or help you code the feature extraction first?

