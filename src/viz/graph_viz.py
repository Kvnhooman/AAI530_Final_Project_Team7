"""Plot the METR-LA sensor network with optional per-sensor error heatmap.

Scatter-plots sensor locations on a lat/lon grid, optionally overlaying
graph edges and coloring nodes by MAE for spatial error analysis.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_sensor_network(
    sensor_locations_path: str,
    adj_matrix: np.ndarray | None = None,
    errors: np.ndarray | None = None,
    save_dir: str = "outputs/figures",
    dpi: int = 150,
) -> None:
    """Plot the sensor network with optional error heatmap overlay.

    When `errors` is provided, nodes are colored by their MAE value using
    a red-yellow-green colormap (RdYlGn_r): green = low error, red = high.
    Graph edges from the adjacency matrix are drawn as faint gray lines to
    show the road network topology.

    Args:
        sensor_locations_path: path to graph_sensor_locations.csv with columns
                               [sensor_id, latitude, longitude]
        adj_matrix:            (N, N) adjacency matrix for drawing edges between sensors
        errors:                (N,) per-sensor error values (e.g., MAE in mph) for
                               coloring nodes — if None, plain blue dots are used
        save_dir:              directory to save the output figure
        dpi:                   resolution of the saved image
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Load sensor coordinates (CSV has no header in METR-LA)
    locs = pd.read_csv(sensor_locations_path)
    cols = locs.columns.tolist()
    if len(cols) >= 3:
        locs.columns = ["sensor_id", "latitude", "longitude"] + cols[3:]

    lat = locs["latitude"].values
    lon = locs["longitude"].values
    n = len(lat)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Draw graph edges as faint gray lines to show road network structure.
    # We only draw the upper triangle (i < j) since the graph is undirected.
    if adj_matrix is not None:
        for i in range(n):
            for j in range(i + 1, n):
                if adj_matrix[i, j] > 0:
                    ax.plot(
                        [lon[i], lon[j]],
                        [lat[i], lat[j]],
                        color="gray",
                        alpha=0.1,
                        linewidth=0.5,
                    )

    # Draw sensor nodes — color by error if available, otherwise plain blue
    if errors is not None:
        sc = ax.scatter(
            lon, lat, c=errors, cmap="RdYlGn_r", s=30, edgecolors="k", linewidths=0.3, zorder=5
        )
        plt.colorbar(sc, ax=ax, label="MAE (mph)", shrink=0.7)
        ax.set_title("METR-LA Sensor Network — Error Heatmap")
    else:
        ax.scatter(lon, lat, c="steelblue", s=30, edgecolors="k", linewidths=0.3, zorder=5)
        ax.set_title("METR-LA Sensor Network")

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fname = "sensor_network.png" if errors is None else "sensor_error_heatmap.png"
    fig.savefig(save_path / fname, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")
