"""Plot training/validation loss, MAE, RMSE, and learning rate curves.

Generates a 2x2 diagnostic grid from the training history dict.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_training_curves(
    history: dict[str, list[float]],
    save_dir: str = "outputs/figures",
    dpi: int = 150,
) -> None:
    """Plot the 2x2 grid of training diagnostics.

    Args:
        history: dict from Trainer.train() with keys:
                 'train_loss', 'val_loss', 'val_mae', 'val_rmse', 'lr'
                 Each value is a list of per-epoch values.
        save_dir: directory to save the figure
        dpi:      image resolution
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Train vs. Val loss — the most important training diagnostic
    axes[0, 0].plot(epochs, history["train_loss"], label="Train")
    axes[0, 0].plot(epochs, history["val_loss"], label="Val")
    axes[0, 0].set_title("Loss (L1)")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Validation MAE — the primary evaluation metric
    axes[0, 1].plot(epochs, history["val_mae"], label="Val MAE", color="orange")
    axes[0, 1].set_title("Validation MAE")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].grid(True, alpha=0.3)

    # Validation RMSE — shows sensitivity to large errors
    axes[1, 0].plot(epochs, history["val_rmse"], label="Val RMSE", color="green")
    axes[1, 0].set_title("Validation RMSE")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].grid(True, alpha=0.3)

    # Learning rate schedule — log scale to see the cosine annealing shape
    axes[1, 1].plot(epochs, history["lr"], label="LR", color="red")
    axes[1, 1].set_title("Learning Rate")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_yscale("log")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path / "training_curves.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
