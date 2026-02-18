"""Preprocess METR-LA HDF5 data into train/val/test sliding windows.

Pipeline: raw HDF5 -> impute missing values -> chronological split ->
Z-score normalize (fitted on train only) -> sliding windows -> save as .npz.
Input features are [speed, time_of_day]; targets are speed-only.
"""

from pathlib import Path

import h5py
import numpy as np


def load_hdf5(hdf5_path: str) -> tuple[np.ndarray, list[str]]:
    """Load speed data from the METR-LA HDF5 file.

    The file uses pandas' HDF5 convention: a 'df' group containing 'axis0'
    (sensor IDs), 'axis1' (timestamps), and 'block0_values' (the speed matrix).

    Returns:
        speeds:     (T, N) float32 array — rows are timesteps, columns are sensors
        timestamps: list of timestamp strings for each row
    """
    with h5py.File(hdf5_path, "r") as f:
        # The METR-LA .h5 was written by pandas.to_hdf(), which stores data
        # under a 'df' group with block storage layout.
        if "df" in f:
            data = f["df"]["block0_values"][:]
            try:
                timestamps = [
                    t.decode() if isinstance(t, bytes) else str(t)
                    for t in f["df"]["axis1"][:]
                ]
            except Exception:
                timestamps = list(range(data.shape[0]))
        else:
            # Fallback for non-pandas HDF5 files
            key = list(f.keys())[0]
            data = f[key][:]
            timestamps = list(range(data.shape[0]))

    return data.astype(np.float32), timestamps


def impute_missing(speeds: np.ndarray) -> np.ndarray:
    """Replace zeros and NaNs with per-sensor mean (simple mean imputation).

    METR-LA encodes missing readings as 0.0 mph (not NaN). We first convert
    those zeros to NaN, then fill each sensor's missing values with that
    sensor's mean speed. If an entire sensor column is missing, we fall back
    to the global mean across all sensors.

    Args:
        speeds: (T, N) raw speed array — zeros indicate missing readings

    Returns:
        (T, N) array with missing values replaced by per-sensor means
    """
    data = speeds.copy()
    data[data == 0.0] = np.nan

    # Per-sensor mean imputation
    sensor_means = np.nanmean(data, axis=0)
    for j in range(data.shape[1]):
        mask = np.isnan(data[:, j])
        data[mask, j] = sensor_means[j]

    # Global fallback for fully-missing sensors
    global_mean = np.nanmean(data)
    data = np.nan_to_num(data, nan=global_mean)
    return data


def add_time_feature(n_timesteps: int) -> np.ndarray:
    """Create a normalized time-of-day feature cycling every 24 hours.

    At 5-minute intervals, there are 288 steps per day (24 * 60 / 5 = 288).
    We normalize to [0, 1) so the feature is on a similar scale to the
    Z-scored speed values. The model uses this to learn diurnal patterns:
    morning rush hour, midday lull, evening rush, overnight quiet.

    Args:
        n_timesteps: total number of timesteps in the dataset

    Returns:
        (n_timesteps,) float32 array cycling 0.0 → ~0.997 every 288 steps
    """
    steps_per_day = 288
    tod = np.arange(n_timesteps) % steps_per_day
    return (tod / steps_per_day).astype(np.float32)


def create_sliding_windows(
    data: np.ndarray,
    time_feature: np.ndarray,
    seq_len: int = 12,
    horizon: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
    """Create overlapping (input, target) windows for sequence-to-sequence training.

    Slides a window of size (seq_len + horizon) across the time axis, producing:
        X[i] = data[i : i+seq_len]                     — what the model sees
        Y[i] = data[i+seq_len : i+seq_len+horizon]     — what it predicts

    The input X has 2 channels per node: [speed, time_of_day].
    The target Y is speed-only (single channel).

    Args:
        data:         (T, N) normalized speed data
        time_feature: (T,) time-of-day feature (0 to 1)
        seq_len:      number of input timesteps  (default 12 = 1 hour)
        horizon:      number of target timesteps (default 12 = 1 hour)

    Returns:
        X: (num_windows, seq_len, N, 2) — input sequences with [speed, time_of_day]
        Y: (num_windows, horizon, N)    — target speed sequences
    """
    T, N = data.shape
    num_windows = T - seq_len - horizon + 1

    X = np.zeros((num_windows, seq_len, N, 2), dtype=np.float32)
    Y = np.zeros((num_windows, horizon, N), dtype=np.float32)

    for i in range(num_windows):
        # Channel 0: normalized speed values for this input window
        X[i, :, :, 0] = data[i : i + seq_len]

        # Channel 1: time-of-day broadcast across all sensors (same time for all)
        X[i, :, :, 1] = time_feature[i : i + seq_len, np.newaxis] * np.ones(N)

        # Target: the next `horizon` timesteps of speed (what we want to predict)
        Y[i] = data[i + seq_len : i + seq_len + horizon]

    return X, Y


def preprocess_and_save(
    hdf5_path: str,
    output_dir: str,
    seq_len: int = 12,
    horizon: int = 12,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
) -> None:
    """Full preprocessing pipeline: load -> impute -> split -> normalize -> window -> save.

    Uses chronological splitting (70/10/20) and fits the Z-score scaler on
    the training split only to prevent data leakage.

    Args:
        hdf5_path:   path to the raw metr-la.h5 file
        output_dir:  where to save processed .npz files
        seq_len:     input window length (default 12 = 1 hour)
        horizon:     prediction horizon  (default 12 = 1 hour)
        train_ratio: fraction of data for training  (0.7)
        val_ratio:   fraction of data for validation (0.1)
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading HDF5 ...")
    speeds, timestamps = load_hdf5(hdf5_path)
    print(f"  Raw shape: {speeds.shape} ({speeds.shape[0]} timesteps, {speeds.shape[1]} sensors)")

    print("Imputing missing values ...")
    speeds = impute_missing(speeds)

    # Chronological split boundaries (computed on raw timesteps, before windowing)
    T = speeds.shape[0]
    train_end = int(T * train_ratio)
    val_end = int(T * (train_ratio + val_ratio))

    # Fit Z-score scaler on training data only to prevent leakage
    train_data = speeds[:train_end]
    mean = train_data.mean(axis=0)  # per-sensor mean: shape (N,)
    std = train_data.std(axis=0)    # per-sensor std:  shape (N,)
    std[std < 1e-6] = 1.0           # guard against division by zero for constant sensors
    print(f"  Scaler — mean range: [{mean.min():.1f}, {mean.max():.1f}], std range: [{std.min():.1f}, {std.max():.1f}]")

    # Apply normalization: (speed - mean) / std for each sensor independently
    speeds_norm = (speeds - mean) / std

    # Build the time-of-day feature for the entire time range
    time_feat = add_time_feature(T)

    # Persist timestamps for later use (streaming replay, Tableau export)
    np.save(out / "timestamps.npy", np.array(timestamps, dtype=object), allow_pickle=True)

    # Create sliding windows for each split, with boundary overlap
    splits = {
        "train": (0, train_end),
        "val": (train_end - seq_len, val_end),     # overlap to keep first val window valid
        "test": (val_end - seq_len, T),             # overlap to keep first test window valid
    }

    for name, (start, end) in splits.items():
        chunk = speeds_norm[start:end]
        tf = time_feat[start:end]
        X, Y = create_sliding_windows(chunk, tf, seq_len, horizon)
        np.savez(out / f"{name}.npz", X=X, Y=Y)
        print(f"  {name}: X={X.shape}, Y={Y.shape}")

    # Save scaler parameters for inverse-transforming predictions back to mph
    np.savez(out / "scaler.npz", mean=mean, std=std)
    print(f"Preprocessing complete → {out.resolve()}")


if __name__ == "__main__":
    preprocess_and_save(
        hdf5_path="data/raw/metr-la.h5",
        output_dir="data/processed",
    )
