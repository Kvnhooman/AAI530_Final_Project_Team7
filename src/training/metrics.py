"""MAE, RMSE, and MAPE metrics with null-value masking.

Masks out zero-speed target values (missing sensor readings) to avoid
distorting accuracy measurements. Uses the normalized-mask approach from
DCRNN / Graph WaveNet for fair comparison with published baselines.
"""

import torch
import numpy as np


def masked_mae(pred: torch.Tensor, target: torch.Tensor, null_val: float = 0.0) -> torch.Tensor:
    """Mean Absolute Error with masking of null values.

    Ignores target values close to null_val (default 0.0) to avoid
    penalizing predictions for invalid sensor readings.

    Args:
        pred:     model predictions (any shape, must match target)
        target:   ground truth values (same shape as pred)
        null_val: value to treat as missing/invalid (default 0.0)

    Returns:
        scalar tensor with the masked MAE value
    """
    # Create mask: True for valid readings, False for null values
    mask = ~torch.isclose(target, torch.tensor(null_val, device=target.device), atol=1e-5)
    mask = mask.float()
    # Normalize mask so that masked-out entries don't reduce the denominator.
    # Without this, having more nulls would artificially lower the metric.
    mask /= torch.mean(mask).clamp(min=1e-8)
    loss = torch.abs(pred - target) * mask
    return torch.mean(loss)


def masked_rmse(pred: torch.Tensor, target: torch.Tensor, null_val: float = 0.0) -> torch.Tensor:
    """Root Mean Squared Error with masking of null values.

    RMSE penalizes large errors more than MAE (due to squaring), making it
    useful for detecting if the model has occasional big misses.

    Args:
        pred:     model predictions
        target:   ground truth values
        null_val: value to treat as missing/invalid

    Returns:
        scalar tensor with the masked RMSE value
    """
    mask = ~torch.isclose(target, torch.tensor(null_val, device=target.device), atol=1e-5)
    mask = mask.float()
    mask /= torch.mean(mask).clamp(min=1e-8)
    loss = ((pred - target) ** 2) * mask
    return torch.sqrt(torch.mean(loss))


def masked_mape(pred: torch.Tensor, target: torch.Tensor, null_val: float = 0.0) -> torch.Tensor:
    """Mean Absolute Percentage Error with masking of null/zero values.

    MAPE expresses error as a percentage of the actual value, making it
    scale-independent. We additionally mask very small target values to
    avoid division-by-near-zero inflating the percentage.

    Returns:
        scalar tensor with the masked MAPE value (in percent, 0-100+)
    """
    mask = ~torch.isclose(target, torch.tensor(null_val, device=target.device), atol=1e-5)
    # Also exclude near-zero targets to avoid division by ~0
    mask = mask & (target.abs() > 1e-5)
    mask = mask.float()
    mask /= torch.mean(mask).clamp(min=1e-8)
    loss = torch.abs((pred - target) / target.clamp(min=1e-5)) * mask
    return torch.mean(loss) * 100.0


def compute_all_metrics(
    pred: np.ndarray, target: np.ndarray
) -> dict[str, float]:
    """Compute MAE, RMSE, and MAPE on numpy arrays (already inverse-transformed to mph).

    This is a convenience function for evaluation scripts that work with
    numpy arrays rather than PyTorch tensors. Internally converts to tensors,
    computes all three metrics, and returns a dictionary.

    Args:
        pred:   numpy array of predictions in mph (any shape)
        target: numpy array of ground truth in mph (same shape as pred)

    Returns:
        dict with 'MAE', 'RMSE', 'MAPE' keys and float values
    """
    pred_t = torch.from_numpy(pred).float()
    target_t = torch.from_numpy(target).float()
    return {
        "MAE": masked_mae(pred_t, target_t).item(),
        "RMSE": masked_rmse(pred_t, target_t).item(),
        "MAPE": masked_mape(pred_t, target_t).item(),
    }
