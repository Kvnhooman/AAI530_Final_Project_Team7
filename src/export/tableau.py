"""Export model predictions as a Tableau-ready star schema.

Generates dimension tables (sensor_dim.csv, time_dim.csv), a fact table
(fact_predictions.parquet), and an edge list (adjacency_edges.csv) that
Tableau can join for interactive dashboard analysis.
"""

from pathlib import Path

import numpy as np
import pandas as pd


def build_sensor_dim(
    sensor_locations_path: str,
    sensor_ids_path: str,
) -> pd.DataFrame:
    """Build the sensor dimension table.

    Maps each sensor's array index (0-206) to its real-world ID and
    geographic coordinates. This table is joined to the fact table
    on sensor_id for geographic filtering and map visualizations.

    Returns:
        DataFrame with columns: sensor_index, sensor_id, latitude, longitude
    """
    locs = pd.read_csv(sensor_locations_path)
    with open(sensor_ids_path) as f:
        sensor_ids = [line.strip() for line in f if line.strip()]

    df = pd.DataFrame(
        {
            "sensor_index": range(len(sensor_ids)),
            "sensor_id": sensor_ids,
        }
    )
    # Merge with location data — the CSV may use a different column name
    locs = locs.rename(columns={"sensor_id": "sensor_id_loc"})
    locs["sensor_id_loc"] = locs["sensor_id_loc"].astype(str)
    df["sensor_id_str"] = df["sensor_id"].astype(str)
    df = df.merge(
        locs, left_on="sensor_id_str", right_on="sensor_id_loc", how="left"
    )
    df = df[["sensor_index", "sensor_id", "latitude", "longitude"]]
    return df


def build_time_dim(timestamps: np.ndarray, start_offset: int = 0) -> pd.DataFrame:
    """Build the time dimension table from the timestamp array.

    Extracts temporal attributes that enable time-based analysis in Tableau:
    hour-of-day patterns, weekday vs. weekend comparisons, etc.

    Args:
        timestamps:   array of timestamp strings from preprocessing
        start_offset: skip this many initial timestamps (for split alignment)

    Returns:
        DataFrame with columns: timestamp, date, hour, minute, day_of_week,
                                day_of_week_num, is_weekend, time_of_day
    """
    ts = pd.to_datetime(timestamps[start_offset:])
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "date": ts.date,
            "hour": ts.hour,
            "minute": ts.minute,
            "day_of_week": ts.day_name(),
            "day_of_week_num": ts.dayofweek,
            "is_weekend": ts.dayofweek >= 5,
            "time_of_day": ts.strftime("%H:%M"),
        }
    )
    return df.drop_duplicates(subset=["timestamp"]).reset_index(drop=True)


def build_fact_predictions(
    preds_mph: np.ndarray,
    targets_mph: np.ndarray,
    sensor_ids: list[str],
    timestamps: np.ndarray,
    seq_len: int = 12,
    split_name: str = "test",
    split_start_idx: int = 0,
) -> pd.DataFrame:
    """Build the fact table with predictions, actuals, and residuals.

    Each row represents one prediction: a specific sensor at a specific
    timestamp and forecast horizon. The table is intentionally denormalized
    (sensor_id and timestamp are repeated) for easy Tableau consumption.

    Args:
        preds_mph:       (num_samples, num_nodes, horizon) predicted speeds in mph
        targets_mph:     (num_samples, num_nodes, horizon) actual speeds in mph
        sensor_ids:      list of sensor ID strings (length = num_nodes)
        timestamps:      full array of timestamps
        seq_len:         input sequence length (for computing target timestamp offset)
        split_name:      which data split ("train", "val", "test")
        split_start_idx: index into timestamps where this split starts

    Returns:
        DataFrame with columns: sensor_id, timestamp, actual_speed_mph,
                                predicted_speed_mph, residual, horizon,
                                horizon_minutes, split
    """
    num_samples, num_nodes, horizon = preds_mph.shape
    rows = []

    for i in range(num_samples):
        for h in range(horizon):
            # The target timestamp for sample i, horizon h:
            # start_idx + seq_len (skip input window) + i (sample offset) + h (horizon step)
            ts_idx = split_start_idx + seq_len + i + h
            if ts_idx >= len(timestamps):
                continue
            ts = timestamps[ts_idx]

            for n in range(num_nodes):
                rows.append(
                    {
                        "sensor_id": sensor_ids[n],
                        "timestamp": ts,
                        "actual_speed_mph": float(targets_mph[i, n, h]),
                        "predicted_speed_mph": float(preds_mph[i, n, h]),
                        "residual": float(preds_mph[i, n, h] - targets_mph[i, n, h]),
                        "horizon": h + 1,
                        "horizon_minutes": (h + 1) * 5,
                        "split": split_name,
                    }
                )

    return pd.DataFrame(rows)


def build_adjacency_edges(
    adj_matrix: np.ndarray,
    sensor_dim: pd.DataFrame,
) -> pd.DataFrame:
    """Build an edge list with coordinates for Tableau network visualization.

    Each row represents one edge in the sensor network with source/target
    coordinates, enabling Tableau to draw the road network as lines on a map.

    Only includes edges in one direction (i < j) since the graph is undirected.

    Args:
        adj_matrix: (N, N) adjacency matrix (from graph.py)
        sensor_dim: sensor dimension table with latitude/longitude

    Returns:
        DataFrame with columns: source_sensor_id, target_sensor_id,
                                source_lat, source_lon, target_lat, target_lon, weight
    """
    rows, cols = np.nonzero(adj_matrix)
    edges = []

    for i, j in zip(rows, cols):
        if i >= j:
            continue  # undirected: only keep one direction
        src = sensor_dim.iloc[i]
        dst = sensor_dim.iloc[j]
        edges.append(
            {
                "source_sensor_id": src["sensor_id"],
                "target_sensor_id": dst["sensor_id"],
                "source_lat": src["latitude"],
                "source_lon": src["longitude"],
                "target_lat": dst["latitude"],
                "target_lon": dst["longitude"],
                "weight": float(adj_matrix[i, j]),
            }
        )

    return pd.DataFrame(edges)


def forecast_to_dashboard_df(
    preds_mph: np.ndarray,
    targets_mph: np.ndarray,
    sensor_ids: list[str],
    timestamps: np.ndarray,
    seq_len: int = 12,
    split_start_idx: int = 0,
) -> pd.DataFrame:
    """Convenience wrapper for building a flat predictions DataFrame.

    Calls build_fact_predictions with split_name="all". Useful when you
    want a single DataFrame for ad-hoc analysis rather than the full
    star-schema export.

    Returns:
        DataFrame suitable for Tableau dashboards or pandas analysis
    """
    return build_fact_predictions(
        preds_mph=preds_mph,
        targets_mph=targets_mph,
        sensor_ids=sensor_ids,
        timestamps=timestamps,
        seq_len=seq_len,
        split_name="all",
        split_start_idx=split_start_idx,
    )


def export_all(
    preds_mph: np.ndarray,
    targets_mph: np.ndarray,
    sensor_locations_path: str,
    sensor_ids_path: str,
    timestamps: np.ndarray,
    adj_matrix: np.ndarray,
    output_dir: str,
    seq_len: int = 12,
    split_name: str = "test",
    split_start_idx: int = 0,
) -> None:
    """Export the complete Tableau star-schema file set.

    Generates four files in the output directory:
        sensor_dim.csv            — sensor dimension table
        time_dim.csv              — time dimension table
        fact_predictions.parquet  — prediction fact table
        adjacency_edges.csv       — network edge list

    Args:
        preds_mph:              (samples, nodes, horizon) predictions in mph
        targets_mph:            (samples, nodes, horizon) actuals in mph
        sensor_locations_path:  path to sensor coordinates CSV
        sensor_ids_path:        path to sensor IDs text file
        timestamps:             full timestamp array
        adj_matrix:             (N, N) adjacency matrix
        output_dir:             where to write the output files
        seq_len:                input sequence length
        split_name:             data split name ("test")
        split_start_idx:        timestamp offset for this split
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(sensor_ids_path) as f:
        sensor_ids = [line.strip() for line in f if line.strip()]

    # Sensor dimension table
    sensor_dim = build_sensor_dim(sensor_locations_path, sensor_ids_path)
    sensor_dim.to_csv(out / "sensor_dim.csv", index=False)
    print(f"  sensor_dim.csv: {len(sensor_dim)} sensors")

    # Time dimension table
    time_dim = build_time_dim(timestamps)
    time_dim.to_csv(out / "time_dim.csv", index=False)
    print(f"  time_dim.csv: {len(time_dim)} unique timestamps")

    # Fact predictions table (parquet for efficiency)
    fact = build_fact_predictions(
        preds_mph, targets_mph, sensor_ids, timestamps,
        seq_len=seq_len, split_name=split_name, split_start_idx=split_start_idx,
    )
    fact.to_parquet(out / "fact_predictions.parquet", index=False)
    print(f"  fact_predictions.parquet: {len(fact):,} rows")

    # Adjacency edge list for network topology visualization
    edges = build_adjacency_edges(adj_matrix, sensor_dim)
    edges.to_csv(out / "adjacency_edges.csv", index=False)
    print(f"  adjacency_edges.csv: {len(edges)} edges")

    print(f"Tableau export complete → {out.resolve()}")
