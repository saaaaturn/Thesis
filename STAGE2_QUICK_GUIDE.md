# Stage 2 Quick Extraction Checklist

## Your Understanding ✅

Yes, correct! You need to extract:

```
From AIG:
├─ Node matrix [254 × F]         ← feature for each gate
├─ Edge matrix [254 × 254]       ← connectivity graph
└─ (Statistics)

From MIG:
├─ Node matrix [N × F]           ← feature for each MAJ gate
├─ Edge matrix [N × N]           ← connectivity graph
└─ (Statistics)

Both:
└─ Statistical characteristics [S] ← global circuit properties
```

---

## Clarification: The "Node Matrix"

**"Node matrix" means:**

```
NOT a matrix like [1, 0, 1, 0, ...] (one row)

INSTEAD: A MATRIX with one row per node/gate:

       gate_type  fanin  fanout  is_critical  depth
Node 0    [  0       2      3        0         1   ]
Node 1    [  0       2      1        1         2   ]
Node 2    [  1       3      1        0         3   ]
Node 3    [  2       1      2        1         4   ]
...
Node 253  [  0       2      0        0        10   ]

Shape: [254 rows × 5 columns] for 256-bit adder AIG
```

Each row = features of one gate
Each column = one type of feature

---

## Clarification: The "Edge Matrix" / "Adjacency"

**"Edge matrix" = Adjacency matrix** representing graph structure:

### Option A: Dense Format (Simple)
```
254 × 254 matrix:
       to:0  to:1  to:2  to:3  ...
from:0 [  0     0     1     0   ]  Gate 0 connects to Gate 2
from:1 [  0     0     1     0   ]  Gate 1 connects to Gate 2
from:2 [  0     0     0     0   ]  Gate 2 connects nowhere
from:3 [  0     0     1     0   ]  Gate 3 connects to Gate 2
...
```

Interpretation: A[i][j] = 1 means gate i outputs to gate j

### Option B: Sparse Format (Efficient - RECOMMENDED)
```
Just store the edges:
[from, to]
[  0,   2 ]
[  1,   2 ]
[  3,   2 ]
...

Much smaller (only ~500 entries instead of 254×254=64K)
```

---

## Statistical Characteristics

Extract ONCE (can use either AIG or MIG or both):

```
statistics = {
  'num_gates': 254,           // Total AND gates
  'num_edges': 500,           // Total connections  
  'max_depth': 51,            // Longest path
  'avg_fanin': 2.0,           // Average inputs per gate
  'avg_fanout': 1.97,         // Average outputs per gate
  'num_inputs': 256,          // Primary inputs (I/O)
  'num_outputs': 129,         // Primary outputs (I/O)
  'critical_path_length': 51, // Longest propagation path
  'is_arithmetic': 1.0,       // Heuristic: circuit type
  'gate_density': 0.015,      // gates / (inputs + outputs)
  // ... add more as needed ...
}
```

---

## Implementation Recommendation

**Start with simple text files for debugging:**

### 1. Node Features File
```
Filename: aig_node_features.txt

0 2 3 0 1
0 2 1 1 2
1 3 1 0 3
0 2 2 1 4
...
```

Format: space-separated values, one node per line
Columns: gate_type fanin fanout is_critical depth

### 2. Adjacency File  
```
Filename: aig_adjacency.txt

0 2
1 2
3 2
...
```

Format: src dst (edge list format - more efficient than dense matrix)

### 3. Statistics File
```
Filename: aig_statistics.txt

num_gates 254
num_edges 500
max_depth 51
avg_fanin 2.0
avg_fanout 1.97
...
```

Format: key value pairs

---

## What Mockturtle Provides

From the AIG/MIG objects, you can extract:

```cpp
// Node count
aig.num_gates()        // How many gates

// Node properties
aig.is_and(node)       // Is it an AND gate?
aig.is_maj(node)       // Is it a majority gate?
aig.is_pi(node)        // Primary input?
aig.is_po(node)        // Primary output?

// Connectivity
aig.fanin_size(node)   // How many inputs
// Fanout: need to count yourself by iterating

// Traversal
aig.foreach_node(...)  // Loop over all nodes
aig.foreach_fanin(...) // Loop over inputs to a node
```

---

## Pseudocode: Complete Extraction

```cpp
// In interface.cpp

#include <fstream>

void extract_and_save_features(const std::string& filename, bool extract_aig) {
  aig_network aig;
  mig_network mig;
  
  // Load circuit
  if (extract_aig) {
    read_aiger(filename, aiger_reader(aig));
    extract_from_network(aig, "aig");
  } else {
    read_aiger(filename, aiger_reader(mig));
    extract_from_network(mig, "mig");
  }
}

template<typename Network>
void extract_from_network(Network& net, std::string prefix) {
  std::ofstream node_file(prefix + "_node_features.txt");
  std::ofstream adj_file(prefix + "_adjacency.txt");
  std::ofstream stat_file(prefix + "_statistics.txt");
  
  // 1. Extract node features
  std::map<auto, int> node_idx;
  int idx = 0;
  
  net.foreach_node([&](auto node) {
    int fanin = net.fanin_size(node);
    int gate_type = net.is_and(node) ? 0 : 3;  // simplification
    
    // For now, use dummy values for fanout, critical, depth
    node_file << gate_type << " " << fanin << " " << 0 << " " 
              << 0 << " " << 0 << "\n";
    
    node_idx[node] = idx++;
  });
  node_file.close();
  
  // 2. Extract adjacency (edge list format)
  net.foreach_node([&](auto target) {
    net.foreach_fanin(target, [&](auto fanin_signal) {
      auto source = fanin_signal.index;
      adj_file << node_idx[source] << " " 
               << node_idx[target] << "\n";
    });
  });
  adj_file.close();
  
  // 3. Extract statistics
  stat_file << "num_gates " << net.num_gates() << "\n";
  stat_file << "num_pis " << net.num_pis() << "\n";
  stat_file << "num_pos " << net.num_pos() << "\n";
  // ... compute and save more stats ...
  stat_file.close();
}
```

---

## Next: What to Do With These

Once extracted, these files serve as **input to GCN**:

1. **Python loads these files:**
   ```python
   import numpy as np
   
   node_features = np.loadtxt('aig_node_features.txt')  # [254 × 5]
   adjacency = np.loadtxt('aig_adjacency.txt', dtype=int)  # [E × 2]
   statistics = load_statistics('aig_statistics.txt')  # dict
   ```

2. **GCN uses them:**
   ```python
   # Message passing
   gcn_output = gnn(node_features, adjacency)  # [254 × 128]
   
   # Pooling to circuit level
   circuit_embedding = global_pool(gcn_output)  # [128]
   ```

3. **Fuse with statistics:**
   ```python
   fused = cross_attention(aig_embedding, mig_embedding, statistics)
   # [256D state ready for policy network]
   ```

---

## Summary

You understood correctly! Here's the full extraction:

✅ **From AIG**:
- Node features [254 × F]
- Adjacency [E pairs or 254×254 matrix]

✅ **From MIG**:
- Node features [N × F]
- Adjacency [E pairs or N×N matrix]

✅ **Statistics** (once, from either or both):
- Circuit metrics [S values]

**Next step**: Code this extraction in interface.cpp or extend import.cpp

Want help implementing the extraction code?

