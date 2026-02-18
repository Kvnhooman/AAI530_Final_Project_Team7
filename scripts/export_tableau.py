"""Export model predictions as Tableau-ready star-schema files.

Usage: python scripts/export_tableau.py --config configs/default.yaml
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import get_dataloaders, load_graph, load_scaler
from src.export.tableau import export_all
from src.models.gcn_lstm import GCNLSTM


def main():
    parser = argparse.ArgumentParser(description="Export Tableau star-schema files")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Device selection
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    processed_dir = cfg["data"]["processed_dir"]
    raw_dir = cfg["data"]["raw_dir"]

    # Load graph, scaler, and test data
    loaders = get_dataloaders(processed_dir, batch_size=cfg["data"]["batch_size"], num_workers=cfg["data"]["num_workers"])
    edge_index, edge_weight = load_graph(str(Path(processed_dir) / "adj_mx.npz"))
    mean, std = load_scaler(str(Path(processed_dir) / "scaler.npz"))

    edge_index_dev = edge_index.to(device)
    edge_weight_dev = edge_weight.to(device)

    # Load adjacency matrix (dense form, for the edge table export)
    adj_data = np.load(str(Path(processed_dir) / "adj_mx.npz"))
    adj_matrix = adj_data["adj_mx"]

    # Load trained model
    model = GCNLSTM(
        in_channels=cfg["model"]["in_channels"],
        hidden_dim=cfg["model"]["hidden_dim"],
        out_horizon=cfg["model"]["out_horizon"],
        num_nodes=cfg["model"]["num_nodes"],
        dropout=cfg["model"]["dropout"],
    ).to(device)

    ckpt_path = Path(cfg["training"]["checkpoint_dir"]) / cfg["training"]["best_model"]
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded model from epoch {checkpoint['epoch']}")

    # Run inference on the test set
    all_preds, all_targets = [], []
    with torch.no_grad():
        for X, Y in loaders["test"]:
            X = X.to(device)
            pred = model(X, edge_index_dev, edge_weight_dev)
            target = Y.permute(0, 2, 1)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(target.numpy())

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    # Inverse transform back to mph
    mean_exp = mean[np.newaxis, :, np.newaxis]
    std_exp = std[np.newaxis, :, np.newaxis]
    preds_orig = preds * std_exp + mean_exp
    targets_orig = targets * std_exp + mean_exp

    # Load timestamps for the time dimension table
    timestamps = np.load(str(Path(processed_dir) / "timestamps.npy"), allow_pickle=True)

    # Export the full star-schema file set
    print("\n── Exporting Tableau files ──")
    export_all(
        preds_mph=preds_orig,
        targets_mph=targets_orig,
        sensor_locations_path=str(Path(raw_dir) / cfg["data"]["sensor_locations_file"]),
        sensor_ids_path=str(Path(raw_dir) / cfg["data"]["sensor_ids_file"]),
        timestamps=timestamps,
        adj_matrix=adj_matrix,
        output_dir=cfg["export"]["output_dir"],
        seq_len=cfg["data"]["seq_len"],
        split_name="test",
    )


if __name__ == "__main__":
    main()
