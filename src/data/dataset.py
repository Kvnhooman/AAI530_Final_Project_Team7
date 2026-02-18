"""PyTorch Dataset and DataLoader wrappers for preprocessed METR-LA data.

Loads .npz sliding windows into Dataset objects and creates DataLoaders
with appropriate batching, shuffling, and pinned memory settings.
Also provides helpers for loading the graph and Z-score scaler.
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class TrafficDataset(Dataset):
    """Wraps a single split's .npz file into a PyTorch Dataset.

    Each sample is one sliding window: an (X, Y) pair where X is the
    input sequence and Y is the target forecast.

    Args:
        npz_path: path to a preprocessed .npz file (train.npz, val.npz, or test.npz)
    """

    def __init__(self, npz_path: str):
        data = np.load(npz_path)
        self.X = torch.from_numpy(data["X"])  # (num_windows, seq_len, num_nodes, 2)
        self.Y = torch.from_numpy(data["Y"])  # (num_windows, horizon, num_nodes)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.Y[idx]


def load_graph(adj_npz_path: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Load the precomputed graph from the .npz file saved by graph.py.

    Returns the COO-format edge_index and edge_weight tensors that
    PyTorch Geometric's GCNConv layer expects.

    Returns:
        edge_index:  (2, E) long tensor — source and target node indices
        edge_weight: (E,) float tensor  — Gaussian kernel edge weights
    """
    data = np.load(adj_npz_path)
    edge_index = torch.from_numpy(data["edge_index"]).long()
    edge_weight = torch.from_numpy(data["edge_weight"]).float()
    return edge_index, edge_weight


def load_scaler(scaler_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load the Z-score normalization parameters saved during preprocessing.

    These are needed to inverse-transform model predictions from normalized
    space back to mph for evaluation and visualization.

    Returns:
        mean: (N,) per-sensor mean speeds from the training split
        std:  (N,) per-sensor standard deviations from the training split
    """
    data = np.load(scaler_path)
    return data["mean"], data["std"]


def get_dataloaders(
    processed_dir: str,
    batch_size: int = 64,
    num_workers: int = 4,
) -> dict[str, DataLoader]:
    """Create train/val/test DataLoaders from the preprocessed .npz files.

    Args:
        processed_dir: directory containing train.npz, val.npz, test.npz
        batch_size:    samples per mini-batch (64 is a good default for traffic GNNs)
        num_workers:   number of subprocess workers for parallel data loading

    Returns:
        dict mapping split name -> DataLoader, e.g. loaders["train"]
    """
    processed = Path(processed_dir)
    loaders = {}

    for split in ["train", "val", "test"]:
        npz_path = processed / f"{split}.npz"
        if not npz_path.exists():
            raise FileNotFoundError(f"Missing {npz_path}. Run preprocessing first.")
        ds = TrafficDataset(str(npz_path))
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),     # only shuffle training data
            num_workers=num_workers,
            pin_memory=True,                # faster CPU->GPU transfers
            drop_last=(split == "train"),   # avoid tiny final batch during training
        )

    return loaders
