"""Actual vs. predicted speed time series overlays.

Generates single-horizon and multi-horizon (5/15/30/60 min) comparison
plots for individual sensors over a 24-hour window.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_sensor_timeseries(
    actual: np.ndarray,
    predicted: np.ndarray,
    sensor_idx: int = 0,
    horizon_idx: int = 0,
    n_steps: int = 288,
    save_dir: str = "outputs/figures",
    dpi: int = 150,
) -> None:
    """Plot actual vs. predicted speed for a single sensor and forecast horizon.

    Args:
        actual:      (samples, nodes, horizon) inverse-transformed actual speeds in mph
        predicted:   (samples, nodes, horizon) inverse-transformed predicted speeds in mph
        sensor_idx:  which sensor to plot (0 to 206)
        horizon_idx: which forecast step to plot (0=5min, 2=15min, 5=30min, 11=60min)
        n_steps:     how many timesteps to show (288 = 1 full day)
        save_dir:    output directory for the figure
        dpi:         image resolution
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Extract the time series for this sensor and horizon
    act = actual[:n_steps, sensor_idx, horizon_idx]
    pred = predicted[:n_steps, sensor_idx, horizon_idx]
    x = np.arange(n_steps) * 5  # convert timestep index to minutes

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(x, act, label="Actual", alpha=0.8, linewidth=1)
    ax.plot(x, pred, label="Predicted", alpha=0.8, linewidth=1)
    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("Speed (mph)")
    ax.set_title(f"Sensor {sensor_idx} — Horizon {(horizon_idx+1)*5} min")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fname = f"timeseries_sensor{sensor_idx}_h{horizon_idx}.png"
    fig.savefig(save_path / fname, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")


def plot_multi_horizon(
    actual: np.ndarray,
    predicted: np.ndarray,
    sensor_idx: int = 0,
    n_steps: int = 288,
    save_dir: str = "outputs/figures",
    dpi: int = 150,
) -> None:
    """Plot actual vs. predicted overlays for 4 forecast horizons (5, 15, 30, 60 min).

    This is the key diagnostic plot: it shows how prediction quality degrades
    as we forecast further into the future. A well-trained model should show:
        - 5 min:  nearly perfect overlap with actual
        - 15 min: very close, slight smoothing
        - 30 min: noticeable smoothing, misses some peaks
        - 60 min: captures the general trend but misses sharp changes

    Args:
        actual:     (samples, nodes, horizon) actual speeds in mph
        predicted:  (samples, nodes, horizon) predicted speeds in mph
        sensor_idx: which sensor to plot
        n_steps:    timesteps to show (288 = 1 day)
        save_dir:   output directory
        dpi:        image resolution
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Four representative horizons: 5 min, 15 min, 30 min, 60 min
    horizons = [(0, "5 min"), (2, "15 min"), (5, "30 min"), (11, "60 min")]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)

    for ax, (h_idx, label) in zip(axes.flat, horizons):
        if h_idx >= actual.shape[2]:
            continue
        act = actual[:n_steps, sensor_idx, h_idx]
        pred = predicted[:n_steps, sensor_idx, h_idx]
        x = np.arange(n_steps) * 5

        ax.plot(x, act, label="Actual", alpha=0.8, linewidth=1)
        ax.plot(x, pred, label="Predicted", alpha=0.8, linewidth=1)
        ax.set_title(f"Horizon: {label}")
        ax.set_ylabel("Speed (mph)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[1, 0].set_xlabel("Time (minutes)")
    axes[1, 1].set_xlabel("Time (minutes)")

    fig.suptitle(f"Sensor {sensor_idx} — Multi-Horizon Forecast", fontsize=13)
    fig.tight_layout()
    fname = f"multi_horizon_sensor{sensor_idx}.png"
    fig.savefig(save_path / fname, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")
