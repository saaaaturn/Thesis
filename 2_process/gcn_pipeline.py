#//////////////////////////////////////////////////////////////////////////////////
#//                     PROCESS: GCN + MLP + Cross-Attention                     //
#//Load extracted matrices, run dual GCN encoders, then fuse with cross-attention//
#//Output final 320D state vector to: GCN_results/<circuit>/<run_tag>/<circuit>_320d.txt //
#//Output embeddings to folder: GCN_results/                                             //
#//////////////////////////////////////////////////////////////////////////////////

#!/usr/bin/env python3
"""Load extracted matrices, run dual GCN encoders, then fuse with cross-attention.

Usage:
  python gcn_pipeline.py --circuit adder
"""

#/////////////////////////////////////////////////////////////////////////////////
#//////////////////////////////////IMPORTS AND DEFINITIONS////////////////////////
#/////////////////////////////////////////////////////////////////////////////////

#import standard libraries
from __future__ import annotations      

import argparse
from pathlib import Path
from typing import Tuple

import torch                    #PyTorch for tensor operations and neural network modules
import torch.nn as nn

#////////////////////////////////////////////////////////////////////////////////
#/////////////////////////////////////GCN + FUSION //////////////////////////////
#////////////////////////////////////////////////////////////////////////////////

#Functions to load node features.
def load_node_features(path: Path) -> torch.Tensor:
    """Load node features from text file, returning tensor [N, 2]: [gate_type, fanin_count]."""
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            node_id, gate_type, fanin_count = map(int, line.split())
            _ = node_id  # Node id is implicit by row order after loading.
            rows.append([float(gate_type), float(fanin_count)])

    if not rows:
        raise ValueError(f"No node features found in {path}")

    x = torch.tensor(rows, dtype=torch.float32)

    # Normalize feature columns to stabilize training.
    x[:, 0] = x[:, 0] / 2.0
    max_fanin = torch.clamp(x[:, 1].max(), min=1.0)
    x[:, 1] = x[:, 1] / max_fanin
    return x

#Functions to load global statistics.
def load_statistics(path: Path) -> torch.Tensor:
    """Load global circuit statistics and return normalized tensor [4].

    Expected keys:
      num_pis, num_pos, num_gates, avg_fanin
    """
    stats = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, value = line.split()
            stats[key] = float(value)

    required = ["num_pis", "num_pos", "num_gates", "avg_fanin"]
    missing = [k for k in required if k not in stats]
    if missing:
        raise ValueError(f"Missing statistics keys in {path}: {missing}")

    # Light normalization to keep scales comparable for MLP input.
    vec = torch.tensor(
        [
            torch.log1p(torch.tensor(stats["num_pis"], dtype=torch.float32)),
            torch.log1p(torch.tensor(stats["num_pos"], dtype=torch.float32)),
            torch.log1p(torch.tensor(stats["num_gates"], dtype=torch.float32)),
            torch.tensor(stats["avg_fanin"], dtype=torch.float32) / 4.0,
        ],
        dtype=torch.float32,
    )
    return vec

#Functions to load edge indices.
def load_edge_index(path: Path) -> torch.Tensor:
    """Load sparse edge list from text file, returning [2, E] with (source, target)."""
    edges = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            source, target = map(int, line.split())
            edges.append((source, target))

    if not edges:
        raise ValueError(f"No edges found in {path}")

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index

#Functions to build normalized adjacency.
def build_norm_adj(num_nodes: int, edge_index: torch.Tensor) -> torch.Tensor:
    """Build row-normalized sparse adjacency with self-loops.

    We aggregate from parent/source nodes into target nodes, matching circuit data flow.
    """
    source = edge_index[0]
    target = edge_index[1]

    # A[target, source] = 1 so each node aggregates incoming messages from its fanins.
    indices = torch.stack([target, source], dim=0)
    values = torch.ones(indices.shape[1], dtype=torch.float32)

    # Add self-loops.
    self_loops = torch.arange(num_nodes, dtype=torch.long)
    self_indices = torch.stack([self_loops, self_loops], dim=0)
    self_values = torch.ones(num_nodes, dtype=torch.float32)

    all_indices = torch.cat([indices, self_indices], dim=1)
    all_values = torch.cat([values, self_values], dim=0)

    adj = torch.sparse_coo_tensor(
        all_indices,
        all_values,
        size=(num_nodes, num_nodes),
        dtype=torch.float32,
    ).coalesce()

    # Row-normalize: D^{-1}A
    row_sum = torch.zeros(num_nodes, dtype=torch.float32)
    row_sum.index_add_(0, adj.indices()[0], adj.values())
    inv_row_sum = torch.where(row_sum > 0, 1.0 / row_sum, torch.zeros_like(row_sum))
    norm_values = adj.values() * inv_row_sum[adj.indices()[0]]

    norm_adj = torch.sparse_coo_tensor(
        adj.indices(),
        norm_values,
        size=(num_nodes, num_nodes),
        dtype=torch.float32,
    ).coalesce()
    return norm_adj

#Simple 2-layer GCN with mean+max pooling and cross-attention fusion.
class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor, norm_adj: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = torch.sparse.mm(norm_adj, x)
        return x

#Shared GCN encoder for AIG and MIG, producing 128D embeddings.
class CircuitGCNEncoder(nn.Module):
    """2-layer GCN encoder producing a fixed 128D embedding for one representation."""

    def __init__(self, in_dim: int = 2, hidden_dim: int = 64, node_dim: int = 128, out_dim: int = 128) -> None:
        super().__init__()
        self.gcn1 = GCNLayer(in_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, node_dim)
        self.act = nn.ReLU()

        # Pool mean + max (2 * node_dim) then project to fixed out_dim (128D).
        self.project = nn.Sequential(
            nn.Linear(2 * node_dim, out_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor, norm_adj: torch.Tensor) -> torch.Tensor:
        h = self.act(self.gcn1(x, norm_adj))
        h = self.act(self.gcn2(h, norm_adj))

        mean_pool = h.mean(dim=0)
        max_pool = h.max(dim=0).values
        graph_vec = torch.cat([mean_pool, max_pool], dim=0)
        return self.project(graph_vec)

#Cross-attention fusion to learn dynamic weights for AIG and MIG embeddings and return fused 256D state.
class CrossAttentionFusion(nn.Module):
    """Learn dynamic weights for AIG and MIG embeddings and return fused 256D state."""

    def __init__(self, embed_dim: int = 128) -> None:
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(2 * embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, aig_embed: torch.Tensor, mig_embed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        joined = torch.cat([aig_embed, mig_embed], dim=0)
        logits = self.scorer(joined)
        weights = torch.softmax(logits, dim=0)

        weighted_aig = weights[0] * aig_embed
        weighted_mig = weights[1] * mig_embed

        fused_256 = torch.cat([weighted_aig, weighted_mig], dim=0)
        return fused_256, weights

#Final state encoder that combines GCN fusion with stats MLP.
class StatsMLP(nn.Module):
    """Encode 4 global statistics into a 64D embedding."""

    def __init__(self, in_dim: int = 4, hidden_dim: int = 64, out_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
        )

    def forward(self, x_stats: torch.Tensor) -> torch.Tensor:
        return self.net(x_stats)

#State encoder
class HybridSYNStateEncoder(nn.Module):
    def __init__(self, stats_dim: int = 64) -> None:
        super().__init__()
        self.aig_encoder = CircuitGCNEncoder()
        self.mig_encoder = CircuitGCNEncoder()
        self.fusion = CrossAttentionFusion(embed_dim=128)
        self.stats_encoder = StatsMLP(in_dim=4, hidden_dim=64, out_dim=stats_dim)

    def forward(
        self,
        x_aig: torch.Tensor,
        adj_aig: torch.Tensor,
        x_mig: torch.Tensor,
        adj_mig: torch.Tensor,
        x_stats: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        aig_embed = self.aig_encoder(x_aig, adj_aig)
        mig_embed = self.mig_encoder(x_mig, adj_mig)
        fused_state, weights = self.fusion(aig_embed, mig_embed)
        stats_embed = self.stats_encoder(x_stats)
        final_state = torch.cat([fused_state, stats_embed], dim=0)
        return aig_embed, mig_embed, fused_state, stats_embed, final_state, weights

#////////////////////////////////////////////////////////////////////////////////
#////////////////////////////////////////MAIN FUNCTION///////////////////////////
#////////////////////////////////////////////////////////////////////////////////

def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 3/4 GCN + fusion pipeline")
    parser.add_argument("--circuit", type=str, default="adder", help="Circuit base name, e.g., adder")
    parser.add_argument(
        "--run-tag",
        type=str,
        default="",
        help="Optional run subfolder name for matrix inputs, e.g., forAIG/<circuit>/<run_tag>/",
    )
    parser.add_argument(
        "--matrix-circuit",
        type=str,
        default="",
        help="Optional matrix folder circuit key; when set, read from forAIG/<matrix_circuit>/<run_tag>/",
    )
    parser.add_argument(
        "--matrix-dir",
        type=str,
        default="matrix",
        help="Directory containing forAIG/ and forMIG/",
    )
    parser.add_argument(
        "--save-state",
        type=str,
        default=None,
        help="Optional explicit output file for final 320D state. If omitted, uses <gcn-results-dir>/<circuit>/<run-tag>/<circuit>_320d.txt",
    )
    parser.add_argument(
        "--gcn-results-dir",
        type=str,
        default="GCN_results",
        help="Directory to save AIG/MIG 128D GCN embeddings",
    )
    parser.add_argument(
        "--stats-dim",
        type=int,
        default=64,
        help="Output dimension of statistics MLP embedding",
    )
    args = parser.parse_args()

    matrix_root = Path(args.matrix_dir)
    matrix_folder = args.matrix_circuit.strip() if args.matrix_circuit else args.circuit
    matrix_subdir = Path(matrix_folder)
    if args.run_tag:
        matrix_subdir = matrix_subdir / args.run_tag

    aig_node_path = matrix_root / "forAIG" / matrix_subdir / f"{args.circuit}_node_features.txt"
    aig_edge_path = matrix_root / "forAIG" / matrix_subdir / f"{args.circuit}_adjacency.txt"
    aig_stats_path = matrix_root / "forAIG" / matrix_subdir / f"{args.circuit}_statistics.txt"
    mig_node_path = matrix_root / "forMIG" / matrix_subdir / f"{args.circuit}_node_features.txt"
    mig_edge_path = matrix_root / "forMIG" / matrix_subdir / f"{args.circuit}_adjacency.txt"

    for p in [aig_node_path, aig_edge_path, aig_stats_path, mig_node_path, mig_edge_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing input file: {p}")

    x_aig = load_node_features(aig_node_path)
    e_aig = load_edge_index(aig_edge_path)
    x_stats = load_statistics(aig_stats_path)
    x_mig = load_node_features(mig_node_path)
    e_mig = load_edge_index(mig_edge_path)

    adj_aig = build_norm_adj(x_aig.shape[0], e_aig)
    adj_mig = build_norm_adj(x_mig.shape[0], e_mig)

    model = HybridSYNStateEncoder(stats_dim=args.stats_dim)
    model.eval()

    with torch.no_grad():
        aig_embed, mig_embed, fused_state, stats_embed, final_state, weights = model(
            x_aig, adj_aig, x_mig, adj_mig, x_stats
        )

    print("=== Stage 3/4 Forward Pass Complete ===")
    print(f"AIG nodes/features: {tuple(x_aig.shape)}")
    print(f"MIG nodes/features: {tuple(x_mig.shape)}")
    print(f"AIG embedding shape: {tuple(aig_embed.shape)}")
    print(f"MIG embedding shape: {tuple(mig_embed.shape)}")
    print(f"Fused graph state shape (cross-attn): {tuple(fused_state.shape)}")
    print(f"Stats embedding shape: {tuple(stats_embed.shape)}")
    print(f"Final state shape (graph + stats): {tuple(final_state.shape)}")
    print(f"Cross-attention weights [AIG, MIG]: {weights.tolist()}")

    gcn_results_dir = Path(args.gcn_results_dir)
    gcn_results_dir.mkdir(parents=True, exist_ok=True)
    output_dir = gcn_results_dir / args.circuit
    if args.run_tag:
        output_dir = output_dir / args.run_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    aig_save_path = gcn_results_dir / f"{args.circuit}_aig_128d.txt"
    mig_save_path = gcn_results_dir / f"{args.circuit}_mig_128d.txt"
    stats_save_path = gcn_results_dir / f"{args.circuit}_stats_{args.stats_dim}d.txt"
    aig_save_path.write_text(" ".join(f"{v:.6f}" for v in aig_embed.tolist()) + "\n", encoding="utf-8")
    mig_save_path.write_text(" ".join(f"{v:.6f}" for v in mig_embed.tolist()) + "\n", encoding="utf-8")
    stats_save_path.write_text(" ".join(f"{v:.6f}" for v in stats_embed.tolist()) + "\n", encoding="utf-8")

    if args.save_state:
        save_path = Path(args.save_state)
    else:
        save_path = output_dir / f"{args.circuit}_320d.txt"
    save_path.write_text(" ".join(f"{v:.6f}" for v in final_state.tolist()) + "\n", encoding="utf-8")
    print(f"Saved AIG 128D embedding to: {aig_save_path}")
    print(f"Saved MIG 128D embedding to: {mig_save_path}")
    print(f"Saved stats {args.stats_dim}D embedding to: {stats_save_path}")
    print(f"Saved final state to: {save_path}")


if __name__ == "__main__":
    main()

#////////////////////////////////////////////////////////////////////////////////
#////////////////////////////////////////END OF FILE/////////////////////////////
#////////////////////////////////////////////////////////////////////////////////