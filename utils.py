import numpy as np
import pandas as pd

def forecast_to_dashboard_df(
    pred,
    node_ids,
    run_time,
    last_obs_time,
    current=None,
    horizon_minutes=5,
    mu=None,
    sigma=None,
    node_expected_mae=None,
    wide=False,
):
    """
    Convert model forecasts into a dashboard-ready DataFrame.

    Parameters
    ----------
    pred : np.ndarray
        Forecast array. Accepted shapes:
        - [N, H]  (rows=node, cols=horizon)
        - [H, N]  (rows=horizon, cols=node)
    node_ids : array-like
        Length N node identifiers in the same order as pred.
    run_time : str or pd.Timestamp
        Timestamp when prediction was generated.
    last_obs_time : str or pd.Timestamp
        Timestamp of the most recent observed value used for prediction.
    current : np.ndarray or None
        Current observed values per node, shape [N]. Optional.
    horizon_minutes : int
        Minutes per step. METR-LA is often 5 minutes.
    mu, sigma : float or None
        If provided, denormalize using: raw = z * sigma + mu.
        Use the SAME mu/sigma you fit on train.
    node_expected_mae : np.ndarray or None
        Expected error per node, shape [N], from validation residuals, used for confidence.
    wide : bool
        If True, returns one row per node with columns pred_t+5, pred_t+10, ...
        If False, returns long format: one row per (node, horizon).

    Returns
    -------
    pd.DataFrame
    """
    run_time = pd.to_datetime(run_time)
    last_obs_time = pd.to_datetime(last_obs_time)

    node_ids = np.asarray(node_ids)
    pred = np.asarray(pred)

    # Normalize pred shape to [N, H]
    N = len(node_ids)
    if pred.ndim != 2:
        raise ValueError(f"pred must be 2D. Got shape {pred.shape}.")
    if pred.shape[0] == N:
        pred_NH = pred
    elif pred.shape[1] == N:
        pred_NH = pred.T
    else:
        raise ValueError(f"pred shape {pred.shape} doesn't match N={N} (node_ids length).")

    H = pred_NH.shape[1]

    # Denormalize if requested
    if (mu is not None) and (sigma is not None):
        pred_raw = pred_NH * float(sigma) + float(mu)
        current_raw = None if current is None else (np.asarray(current) * float(sigma) + float(mu))
    else:
        pred_raw = pred_NH
        current_raw = None if current is None else np.asarray(current)

    # Build long df: one row per (node, horizon)
    horizons = np.arange(1, H + 1)
    horizon_min = horizons * horizon_minutes
    forecast_times = last_obs_time + pd.to_timedelta(horizon_min, unit="m")

    df = pd.DataFrame({
        "node_id": np.repeat(node_ids, H),
        "run_time": run_time,
        "last_obs_time": last_obs_time,
        "horizon_step": np.tile(horizons, N),
        "horizon_minutes": np.tile(horizon_min, N),
        "forecast_time": np.tile(forecast_times.to_numpy(), N),
        "pred": pred_raw.reshape(-1),
    })

    if current_raw is not None:
        df["current"] = np.repeat(current_raw, H)
        df["delta"] = df["pred"] - df["current"]

    # Confidence buckets from expected MAE (optional)
    if node_expected_mae is not None:
        node_expected_mae = np.asarray(node_expected_mae)
        if node_expected_mae.shape[0] != N:
            raise ValueError("node_expected_mae must be length N to match node_ids.")
        df["expected_mae_node"] = np.repeat(node_expected_mae, H)

        # Simple buckets: lower expected error = higher confidence
        q1, q2 = np.quantile(node_expected_mae, [0.33, 0.66])
        def bucket(x):
            if x <= q1:
                return "high"
            elif x <= q2:
                return "medium"
            return "low"
        df["confidence"] = df["expected_mae_node"].map(bucket)

    if not wide:
        return df.sort_values(["node_id", "horizon_step"]).reset_index(drop=True)

    # Wide format (one row per node)
    pivot_cols = {
        f"pred_t+{m}": df.loc[df["horizon_minutes"] == m, "pred"].to_numpy()
        for m in horizon_min
    }
    wide_df = pd.DataFrame({"node_id": node_ids})
    wide_df["run_time"] = run_time
    wide_df["last_obs_time"] = last_obs_time
    if current_raw is not None:
        wide_df["current"] = current_raw

    for m in horizon_min:
        wide_df[f"pred_t+{m}"] = pivot_cols[f"pred_t+{m}"]

        if current_raw is not None:
            wide_df[f"delta_t+{m}"] = wide_df[f"pred_t+{m}"] - wide_df["current"]

    if node_expected_mae is not None:
        wide_df["expected_mae_node"] = node_expected_mae
        q1, q2 = np.quantile(node_expected_mae, [0.33, 0.66])
        wide_df["confidence"] = np.where(
            node_expected_mae <= q1, "high",
            np.where(node_expected_mae <= q2, "medium", "low")
        )

    return wide_df.reset_index(drop=True)
