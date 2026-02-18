"""Training loop with early stopping, checkpointing, and cosine annealing LR.

Trains the GCN+LSTM model using L1 loss (MAE), validates each epoch, saves the
best checkpoint, and stops early if validation loss plateaus.
"""

from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.training.metrics import masked_mae, masked_rmse


class Trainer:
    """Manages model training, validation, early stopping, and checkpointing.

    Args:
        model:           the GCN+LSTM model to train
        edge_index:      (2, E) graph connectivity tensor
        edge_weight:     (E,) edge weight tensor
        device:          torch.device (cuda, mps, or cpu)
        lr:              initial learning rate for Adam (0.003)
        weight_decay:    L2 regularization strength (0.0001)
        epochs:          maximum number of training epochs (100)
        patience:        early stopping patience — epochs without improvement (15)
        grad_clip:       maximum gradient norm for clipping (5.0)
        checkpoint_dir:  where to save model checkpoints
        best_model_name: filename for the best checkpoint
    """

    def __init__(
        self,
        model: nn.Module,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        device: torch.device,
        lr: float = 0.003,
        weight_decay: float = 0.0001,
        epochs: int = 100,
        patience: int = 15,
        grad_clip: float = 5.0,
        checkpoint_dir: str = "outputs/checkpoints",
        best_model_name: str = "best_model.pt",
    ):
        self.model = model.to(device)
        self.edge_index = edge_index.to(device)
        self.edge_weight = edge_weight.to(device)
        self.device = device

        self.epochs = epochs
        self.patience = patience
        self.grad_clip = grad_clip

        self.optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs)
        self.criterion = nn.L1Loss()

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_model_path = self.checkpoint_dir / best_model_name

        self.history = {"train_loss": [], "val_loss": [], "val_mae": [], "val_rmse": [], "lr": []}

    def _train_epoch(self, loader: DataLoader) -> float:
        """Run one training epoch. Returns average loss."""
        self.model.train()
        total_loss = 0.0

        for X, Y in loader:
            X = X.to(self.device)
            Y = Y.to(self.device)

            target = Y.permute(0, 2, 1)  # -> (batch, num_nodes, horizon)

            self.optimizer.zero_grad()
            pred = self.model(X, self.edge_index, self.edge_weight)
            loss = self.criterion(pred, target)
            loss.backward()

            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.optimizer.step()
            total_loss += loss.item() * X.size(0)

        return total_loss / len(loader.dataset)

    @torch.no_grad()
    def _validate(self, loader: DataLoader) -> dict[str, float]:
        """Evaluate on validation set. Returns dict with loss, mae, rmse."""
        self.model.eval()
        total_loss = 0.0
        all_preds, all_targets = [], []

        for X, Y in loader:
            X = X.to(self.device)
            Y = Y.to(self.device)
            target = Y.permute(0, 2, 1)

            pred = self.model(X, self.edge_index, self.edge_weight)
            loss = self.criterion(pred, target)
            total_loss += loss.item() * X.size(0)

            all_preds.append(pred)
            all_targets.append(target)

        preds = torch.cat(all_preds)
        targets = torch.cat(all_targets)

        return {
            "loss": total_loss / len(loader.dataset),
            "mae": masked_mae(preds, targets).item(),
            "rmse": masked_rmse(preds, targets).item(),
        }

    def train(
        self, train_loader: DataLoader, val_loader: DataLoader
    ) -> dict[str, list[float]]:
        """Run the full training loop with early stopping and checkpointing."""
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, self.epochs + 1):
            train_loss = self._train_epoch(train_loader)
            val_metrics = self._validate(val_loader)
            self.scheduler.step()

            lr = self.optimizer.param_groups[0]["lr"]
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_mae"].append(val_metrics["mae"])
            self.history["val_rmse"].append(val_metrics["rmse"])
            self.history["lr"].append(lr)

            print(
                f"Epoch {epoch:3d}/{self.epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val MAE: {val_metrics['mae']:.4f} | "
                f"Val RMSE: {val_metrics['rmse']:.4f} | "
                f"LR: {lr:.6f}"
            )

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                patience_counter = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_loss": best_val_loss,
                        "history": self.history,
                    },
                    self.best_model_path,
                )
                print(f"  ✓ Best model saved (val_loss={best_val_loss:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch} (patience={self.patience})")
                    break

        return self.history
