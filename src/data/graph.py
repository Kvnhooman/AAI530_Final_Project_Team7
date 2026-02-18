"""Build the sensor adjacency matrix from pairwise road-network distances.

Applies a Gaussian kernel W_ij = exp(-d^2 / sigma^2) with epsilon thresholding
to convert raw distances into a sparse weighted graph, then exports it in
PyTorch Geometric's COO format (edge_index + edge_weight).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch


def load_distances(distances_path: str) -> pd.DataFrame:
    """Load the pairwise sensor distances CSV.

    The CSV has three columns: source sensor ID, destination sensor ID,
    and the road-network distance between them (in miles). Not all pairs
    are present — only sensors within a certain radius of each other.

    Returns:
        DataFrame with columns ['from', 'to', 'distance']
    """
    df = pd.read_csv(distances_path)
    df.columns = ["from", "to", "distance"]
    return df


def load_sensor_ids(sensor_ids_path: str) -> list[str]:
    """Load the ordered list of sensor IDs from text file.

    This file defines the canonical ordering: line i corresponds to
    node index i in the adjacency matrix. The 207 METR-LA sensors
    are identified by numeric IDs like '773869', '767541', etc.

    Returns:
        list of 207 sensor ID strings in the canonical graph order
    """
    with open(sensor_ids_path) as f:
        return [line.strip() for line in f if line.strip()]


def build_adjacency_matrix(
    distances_path: str,
    sensor_ids_path: str,
    sigma2: float = 10.0,
    epsilon: float = 0.5,
    include_self_loops: bool = True,
) -> np.ndarray:
    """Build a thresholded Gaussian-kernel weighted adjacency matrix.

    For each pair of sensors (i, j) with distance d_ij:
        W_ij = exp(-d_ij^2 / sigma^2)   if this value >= epsilon
        W_ij = 0                          otherwise

    The resulting matrix is symmetric (undirected graph) because road
    distances are symmetric, and we set both W[i,j] and W[j,i].

    Args:
        distances_path:     path to distances_la_2012.csv
        sensor_ids_path:    path to graph_sensor_ids.txt
        sigma2:             Gaussian kernel bandwidth (controls edge reach)
        epsilon:            minimum weight to keep an edge (sparsification)
        include_self_loops: whether to add 1.0 on the diagonal

    Returns:
        W: (N, N) float32 adjacency matrix where N=207 sensors
    """
    sensor_ids = load_sensor_ids(sensor_ids_path)
    id_to_idx = {int(sid): i for i, sid in enumerate(sensor_ids)}
    n = len(sensor_ids)

    W = np.zeros((n, n), dtype=np.float32)
    distances_df = load_distances(distances_path)

    for _, row in distances_df.iterrows():
        src, dst = int(row["from"]), int(row["to"])

        # Skip sensors not in our canonical list
        if src not in id_to_idx or dst not in id_to_idx:
            continue

        i, j = id_to_idx[src], id_to_idx[dst]

        # Gaussian kernel: closer sensors get higher weights
        w = np.exp(-(row["distance"] ** 2) / sigma2)

        # Only keep edges above the sparsification threshold
        if w >= epsilon:
            W[i, j] = w
            W[j, i] = w  # symmetric because distances are symmetric

    if include_self_loops:
        np.fill_diagonal(W, 1.0)

    return W


def adjacency_to_edge_index(W: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a dense adjacency matrix to PyTorch Geometric's COO sparse format.

    PyG's GCNConv layer expects:
        edge_index: (2, E) long tensor — source and target node indices
        edge_weight: (E,) float tensor — weight for each edge

    We extract these from the nonzero entries of the adjacency matrix.
    The resulting edge_index includes both directions (i->j and j->i)
    plus self-loops, which is what GCNConv needs.

    Args:
        W: (N, N) dense adjacency matrix

    Returns:
        edge_index:  (2, E) tensor of [source_nodes, target_nodes]
        edge_weight: (E,) tensor of corresponding edge weights
    """
    rows, cols = np.nonzero(W)
    edge_index = torch.tensor(np.stack([rows, cols]), dtype=torch.long)
    edge_weight = torch.tensor(W[rows, cols], dtype=torch.float32)
    return edge_index, edge_weight


def build_and_save_graph(
    distances_path: str,
    sensor_ids_path: str,
    output_path: str,
    sigma2: float = 10.0,
    epsilon: float = 0.5,
    include_self_loops: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the adjacency matrix, convert to COO, and persist as .npz.

    Saves three arrays:
        adj_mx:     (N, N) dense adjacency (useful for visualization)
        edge_index: (2, E) COO source/target indices (used by GCNConv)
        edge_weight: (E,) edge weights (used by GCNConv)

    Args:
        distances_path:     path to distances CSV
        sensor_ids_path:    path to sensor IDs text file
        output_path:        where to save the .npz file
        sigma2:             Gaussian kernel bandwidth
        epsilon:            sparsification threshold
        include_self_loops: whether to add self-loops to the graph

    Returns:
        edge_index, edge_weight as PyTorch tensors (also saved to disk)
    """
    W = build_adjacency_matrix(
        distances_path, sensor_ids_path, sigma2, epsilon, include_self_loops
    )
    edge_index, edge_weight = adjacency_to_edge_index(W)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        adj_mx=W,
        edge_index=edge_index.numpy(),
        edge_weight=edge_weight.numpy(),
    )
    print(f"Graph saved to {out} — {edge_index.shape[1]} edges, {W.shape[0]} nodes")
    return edge_index, edge_weight
