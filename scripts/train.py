"""Train the GCN+LSTM traffic forecasting model.

Usage: python scripts/train.py --config configs/default.yaml
"""

import argparse
import sys
from pathlib import Path

import torch
import yaml

# Add project root to Python path so we can import from src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import get_dataloaders, load_graph
from src.models.gcn_lstm import GCNLSTM
from src.training.trainer import Trainer
from src.viz.training_curves import plot_training_curves


def main():
    parser = argparse.ArgumentParser(description="Train GCN+LSTM traffic forecasting model")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Build adjacency matrix if not cached
    adj_path = Path(cfg["data"]["processed_dir"]) / "adj_mx.npz"
    if not adj_path.exists():
        print("Building adjacency matrix ...")
        from src.data.graph import build_and_save_graph

        build_and_save_graph(
            distances_path=str(Path(cfg["data"]["raw_dir"]) / cfg["data"]["distances_file"]),
            sensor_ids_path=str(Path(cfg["data"]["raw_dir"]) / cfg["data"]["sensor_ids_file"]),
            output_path=str(adj_path),
            sigma2=cfg["graph"]["sigma2"],
            epsilon=cfg["graph"]["epsilon"],
            include_self_loops=cfg["graph"]["include_self_loops"],
        )

    edge_index, edge_weight = load_graph(str(adj_path))
    print(f"Graph: {edge_index.shape[1]} edges")

    loaders = get_dataloaders(
        cfg["data"]["processed_dir"],
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
    )
    print(f"DataLoaders: train={len(loaders['train'].dataset)}, val={len(loaders['val'].dataset)}, test={len(loaders['test'].dataset)}")

    model = GCNLSTM(
        in_channels=cfg["model"]["in_channels"],
        hidden_dim=cfg["model"]["hidden_dim"],
        out_horizon=cfg["model"]["out_horizon"],
        num_nodes=cfg["model"]["num_nodes"],
        dropout=cfg["model"]["dropout"],
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    trainer = Trainer(
        model=model,
        edge_index=edge_index,
        edge_weight=edge_weight,
        device=device,
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
        epochs=cfg["training"]["epochs"],
        patience=cfg["training"]["patience"],
        grad_clip=cfg["training"]["grad_clip"],
        checkpoint_dir=cfg["training"]["checkpoint_dir"],
        best_model_name=cfg["training"]["best_model"],
    )

    history = trainer.train(loaders["train"], loaders["val"])

    fig_dir = cfg["viz"]["figure_dir"]
    plot_training_curves(history, save_dir=fig_dir)
    print(f"Training curves saved to {fig_dir}/")


if __name__ == "__main__":
    main()
