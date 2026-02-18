"""Streaming replay engine for simulating real-time traffic predictions.

Replays the test set one window at a time, running model inference and
writing enriched DataFrames (with sensor IDs, timestamps, coordinates)
to the StreamWriter for dashboard consumption.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.dataset import load_graph, load_scaler
from src.models.gcn_lstm import GCNLSTM
from src.streaming.writer import StreamWriter


class StreamSimulator:
    """Replays test set predictions one window at a time, simulating a live feed.

    Args:
        model:            trained GCN+LSTM model (will be set to eval mode)
        edge_index:       (2, E) graph connectivity
        edge_weight:      (E,) edge weights
        scaler_mean:      (N,) per-sensor mean from training set
        scaler_std:       (N,) per-sensor std from training set
        timestamps:       full array of timestamps (for labeling predictions)
        sensor_ids:       list of sensor ID strings (for enrichment)
        sensor_locations: DataFrame with sensor_id, latitude, longitude (optional)
        device:           torch device for inference
    """

    def __init__(
        self,
        model: GCNLSTM,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        scaler_mean: np.ndarray,
        scaler_std: np.ndarray,
        timestamps: np.ndarray,
        sensor_ids: list[str],
        sensor_locations: pd.DataFrame | None = None,
        device: torch.device = torch.device("cpu"),
    ):
        self.model = model.to(device)
        self.model.eval()
        self.edge_index = edge_index.to(device)
        self.edge_weight = edge_weight.to(device)
        self.scaler_mean = scaler_mean
        self.scaler_std = scaler_std
        self.timestamps = timestamps
        self.sensor_ids = sensor_ids
        self.sensor_locations = sensor_locations
        self.device = device

    def _inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Convert normalized predictions back to mph using the training scaler.

        Handles both 2D (nodes, horizon) and 3D (batch, nodes, horizon) inputs
        by broadcasting the scaler arrays appropriately.
        """
        if data.ndim == 2:
            # Single sample: (nodes, horizon) — expand scaler along horizon axis
            return data * self.scaler_std[:, np.newaxis] + self.scaler_mean[:, np.newaxis]
        # Batch: (batch, nodes, horizon) — expand scaler along batch and horizon axes
        mean_exp = self.scaler_mean[np.newaxis, :, np.newaxis]
        std_exp = self.scaler_std[np.newaxis, :, np.newaxis]
        return data * std_exp + mean_exp

    def _build_enriched_df(
        self,
        pred: np.ndarray,
        target: np.ndarray,
        base_ts_idx: int,
        seq_len: int,
    ) -> pd.DataFrame:
        """Build an enriched DataFrame for a single prediction window.

        Each row represents one (sensor, horizon) prediction:
            sensor_id, timestamp, actual_speed_mph, predicted_speed_mph,
            residual, abs_error, horizon_step, horizon_min,
            latitude, longitude (if available)

        Args:
            pred:         (nodes, horizon) inverse-transformed predicted speeds
            target:       (nodes, horizon) inverse-transformed actual speeds
            base_ts_idx:  index into self.timestamps for the start of this window
            seq_len:      input sequence length (predictions start at base + seq_len)
        """
        n_nodes, horizon = pred.shape
        rows = []

        for h in range(horizon):
            # The prediction for horizon h corresponds to the timestamp at
            # base_ts_idx + seq_len + h (i.e., seq_len steps into the future
            # from the start of the input window, plus h more steps)
            ts_idx = base_ts_idx + seq_len + h
            if ts_idx >= len(self.timestamps):
                continue
            ts = self.timestamps[ts_idx]

            for n in range(n_nodes):
                row = {
                    "sensor_id": int(self.sensor_ids[n]),
                    "timestamp": ts,
                    "actual_speed_mph": float(target[n, h]),
                    "predicted_speed_mph": float(pred[n, h]),
                    "residual": float(pred[n, h] - target[n, h]),
                    "abs_error": float(abs(pred[n, h] - target[n, h])),
                    "horizon_step": h + 1,
                    "horizon_min": (h + 1) * 5,
                }

                # Optionally add geographic coordinates for map visualizations
                if self.sensor_locations is not None:
                    loc_row = self.sensor_locations[
                        self.sensor_locations["sensor_id"] == int(self.sensor_ids[n])
                    ]
                    if not loc_row.empty:
                        row["latitude"] = float(loc_row.iloc[0]["latitude"])
                        row["longitude"] = float(loc_row.iloc[0]["longitude"])

                rows.append(row)

        return pd.DataFrame(rows)

    def run(
        self,
        test_X: np.ndarray,
        test_Y: np.ndarray,
        writer: StreamWriter,
        mode: str = "fast",
        interval_sec: float = 5.0,
        seq_len: int = 12,
        test_start_idx: int = 0,
    ) -> None:
        """Run the streaming simulation over all test windows.

        Processes test samples sequentially (not batched) to simulate how a
        real-time system would receive and process data one window at a time.

        Args:
            test_X:         (samples, seq_len, nodes, features) test input windows
            test_Y:         (samples, horizon, nodes) test targets
            writer:         StreamWriter instance for persisting results
            mode:           "fast" (no delay) or "realtime" (pause between steps)
            interval_sec:   seconds to wait between steps in realtime mode
            seq_len:        input sequence length (for timestamp offset calculation)
            test_start_idx: offset into the timestamps array for the test split
        """
        n_samples = test_X.shape[0]
        print(f"Streaming {n_samples} test windows (mode={mode}) ...")

        for i in range(n_samples):
            # Process one window at a time (batch_size=1)
            X = torch.from_numpy(test_X[i : i + 1]).to(self.device)

            with torch.no_grad():
                pred = self.model(X, self.edge_index, self.edge_weight)

            # Extract predictions and targets for this window
            pred_np = pred.cpu().numpy()[0]  # (nodes, horizon)
            target_np = test_Y[i].T          # (horizon, nodes) → (nodes, horizon)

            # Convert from normalized space back to mph
            pred_orig = self._inverse_transform(pred_np)
            target_orig = self._inverse_transform(target_np)

            # Build the enriched DataFrame and write to disk
            df = self._build_enriched_df(
                pred_orig, target_orig, base_ts_idx=test_start_idx + i, seq_len=seq_len
            )
            writer.write(df)

            if (i + 1) % 100 == 0 or i == n_samples - 1:
                print(f"  Step {i + 1}/{n_samples} — {writer.total_written} total rows written")

            # In realtime mode, pause between windows to simulate a live feed
            if mode == "realtime":
                time.sleep(interval_sec)

        print(f"Streaming complete — {writer.total_written} total rows.")
