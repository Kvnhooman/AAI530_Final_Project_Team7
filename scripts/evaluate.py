"""Evaluate a trained model on the test set and generate diagnostic plots.

Computes overall and per-horizon metrics (MAE, RMSE, MAPE), generates
time series overlays and a sensor network error heatmap.

Usage: python scripts/evaluate.py --config configs/default.yaml
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import get_dataloaders, load_graph, load_scaler
from src.models.gcn_lstm import GCNLSTM
from src.training.metrics import compute_all_metrics
from src.viz.timeseries import plot_multi_horizon, plot_sensor_timeseries
from src.viz.graph_viz import plot_sensor_network


def inverse_transform(data: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Convert Z-score normalized predictions back to mph.

    Reverses the normalization applied during preprocessing:
        original = normalized * std + mean

    The broadcasting expands per-sensor mean/std (shape N) to match the
    3D prediction array (samples, nodes, horizon):
        mean[np.newaxis, :, np.newaxis] → (1, N, 1) broadcasts to (S, N, H)

    Args:
        data: (samples, nodes, horizon) normalized predictions or targets
        mean: (N,) per-sensor mean from the training split
        std:  (N,) per-sensor std from the training split

    Returns:
        (samples, nodes, horizon) array in original mph scale
    """
    return data * std[np.newaxis, :, np.newaxis] + mean[np.newaxis, :, np.newaxis]


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained GCN+LSTM model")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the graph structure and normalization scaler
    adj_path = Path(cfg["data"]["processed_dir"]) / "adj_mx.npz"
    edge_index, edge_weight = load_graph(str(adj_path))
    mean, std = load_scaler(str(Path(cfg["data"]["processed_dir"]) / "scaler.npz"))

    # Load test DataLoader
    loaders = get_dataloaders(
        cfg["data"]["processed_dir"],
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
    )

    # Load the trained model from the best checkpoint
    model = GCNLSTM(
        in_channels=cfg["model"]["in_channels"],
        hidden_dim=cfg["model"]["hidden_dim"],
        out_horizon=cfg["model"]["out_horizon"],
        num_nodes=cfg["model"]["num_nodes"],
        dropout=cfg["model"]["dropout"],
    )
    ckpt_path = Path(cfg["training"]["checkpoint_dir"]) / cfg["training"]["best_model"]
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} (val_loss={checkpoint['val_loss']:.4f})")

    # --- Run inference on the full test set ---
    edge_index_dev = edge_index.to(device)
    edge_weight_dev = edge_weight.to(device)

    all_preds, all_targets = [], []
    with torch.no_grad():
        for X, Y in loaders["test"]:
            X = X.to(device)
            pred = model(X, edge_index_dev, edge_weight_dev)  # (B, N, H)
            target = Y.permute(0, 2, 1)  # (B, horizon, N) → (B, N, H)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(target.numpy())

    preds = np.concatenate(all_preds, axis=0)   # (samples, nodes, horizon)
    targets = np.concatenate(all_targets, axis=0)

    # Inverse transform to original mph scale for interpretable metrics
    preds_mph = inverse_transform(preds, mean, std)
    targets_mph = inverse_transform(targets, mean, std)

    # --- Overall test metrics ---
    print("\n=== Overall Test Metrics ===")
    overall = compute_all_metrics(preds_mph, targets_mph)
    for k, v in overall.items():
        print(f"  {k}: {v:.4f}")

    # --- Per-horizon metrics ---
    # Shows how accuracy degrades as we forecast further into the future.
    # Expect: 5-min MAE ~2.5 mph, 60-min MAE ~4.5 mph (typical for METR-LA)
    print("\n=== Per-Horizon Metrics ===")
    print(f"{'Horizon':>10} {'Minutes':>8} {'MAE':>8} {'RMSE':>8} {'MAPE':>8}")
    print("-" * 50)
    for h in range(preds_mph.shape[2]):
        metrics = compute_all_metrics(preds_mph[:, :, h], targets_mph[:, :, h])
        print(f"  {h + 1:>7} {(h + 1) * 5:>7} {metrics['MAE']:>8.3f} {metrics['RMSE']:>8.3f} {metrics['MAPE']:>7.2f}%")

    # --- Visualization ---
    fig_dir = cfg["viz"]["figure_dir"]

    # Single-sensor time series overlay (sensor 0, horizon 15 min)
    plot_sensor_timeseries(targets_mph, preds_mph, sensor_idx=0, horizon_idx=2, save_dir=fig_dir)

    # Multi-horizon comparison (5, 15, 30, 60 min forecasts for sensor 0)
    plot_multi_horizon(targets_mph, preds_mph, sensor_idx=0, save_dir=fig_dir)
    print(f"\nTime series plots saved to {fig_dir}/")

    # Per-sensor MAE heatmap on the LA sensor network map
    # Average MAE across all test samples and horizons for each sensor
    sensor_mae = np.mean(np.abs(preds_mph - targets_mph), axis=(0, 2))  # (num_nodes,)
    sensor_locs_path = Path(cfg["data"]["raw_dir"]) / cfg["data"]["sensor_locations_file"]
    if sensor_locs_path.exists():
        adj_data = np.load(adj_path)
        plot_sensor_network(
            str(sensor_locs_path),
            errors=sensor_mae,
            adj_matrix=adj_data["adj_mx"],
            save_dir=fig_dir,
        )
        print(f"Sensor network plot saved to {fig_dir}/")


if __name__ == "__main__":
    main()
