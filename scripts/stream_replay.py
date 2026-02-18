"""Replay test-set predictions in streaming mode for dashboard integration.

Supports --mode fast (batch) and --mode realtime (timed intervals).
Outputs: cumulative Parquet + rolling CSV (most recent 2000 rows).

Usage: python scripts/stream_replay.py --config configs/default.yaml --mode fast
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import load_graph, load_scaler
from src.models.gcn_lstm import GCNLSTM
from src.streaming.simulator import StreamSimulator
from src.streaming.writer import StreamWriter


def main():
    parser = argparse.ArgumentParser(description="Stream replay of traffic predictions")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--mode", type=str, default="fast", choices=["fast", "realtime"])
    parser.add_argument("--interval", type=float, default=5.0, help="Seconds between steps (realtime mode)")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--reset", action="store_true", help="Clear existing streaming outputs")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Device selection: prefer CUDA, then Apple MPS, then CPU
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

    # Load the graph and scaler (same ones used during training)
    edge_index, edge_weight = load_graph(str(Path(processed_dir) / "adj_mx.npz"))
    mean, std = load_scaler(str(Path(processed_dir) / "scaler.npz"))

    # Load the test set sliding windows directly (not via DataLoader,
    # since we process them one at a time in streaming order)
    test_data = np.load(str(Path(processed_dir) / "test.npz"))
    test_X = test_data["X"]
    test_Y = test_data["Y"]

    # Load metadata for enriching predictions with timestamps and locations
    timestamps = np.load(str(Path(processed_dir) / "timestamps.npy"), allow_pickle=True)

    with open(str(Path(raw_dir) / cfg["data"]["sensor_ids_file"])) as f:
        sensor_ids = [line.strip() for line in f if line.strip()]

    # Load sensor lat/lon for geo-enrichment (optional — works without it)
    loc_path = Path(raw_dir) / cfg["data"]["sensor_locations_file"]
    sensor_locations = None
    if loc_path.exists():
        sensor_locations = pd.read_csv(loc_path)
        # Keep only the columns we need (drop 'index' if present)
        sensor_locations = sensor_locations[["sensor_id", "latitude", "longitude"]]

    # Load the trained model from checkpoint
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
    print(f"Loaded model from epoch {checkpoint['epoch']}")

    # Calculate where the test split starts in the full timestamp array.
    # We need this offset to assign correct timestamps to predictions.
    total_timesteps = len(timestamps)
    test_start_idx = int(total_timesteps * (cfg["data"]["train_ratio"] + cfg["data"]["val_ratio"])) - cfg["data"]["seq_len"]

    # Set up the output writer (parquet + rolling CSV)
    writer = StreamWriter(
        output_dir=cfg["streaming"]["output_dir"],
        parquet_file=cfg["streaming"]["parquet_file"],
        latest_csv=cfg["streaming"]["latest_csv"],
        max_latest_rows=cfg["streaming"]["max_latest_rows"],
    )

    if args.reset:
        writer.reset()
        print("Cleared existing streaming outputs.")

    # Set up the simulator (wraps model inference + inverse transform + enrichment)
    simulator = StreamSimulator(
        model=model,
        edge_index=edge_index,
        edge_weight=edge_weight,
        scaler_mean=mean,
        scaler_std=std,
        timestamps=timestamps,
        sensor_ids=sensor_ids,
        sensor_locations=sensor_locations,
        device=device,
    )

    # Run the streaming simulation
    simulator.run(
        test_X=test_X,
        test_Y=test_Y,
        writer=writer,
        mode=args.mode,
        interval_sec=args.interval,
        seq_len=cfg["data"]["seq_len"],
        test_start_idx=test_start_idx,
    )

    print(f"\nOutputs:")
    print(f"  Parquet: {writer.parquet_path}")
    print(f"  Latest CSV: {writer.csv_path}")


if __name__ == "__main__":
    main()
