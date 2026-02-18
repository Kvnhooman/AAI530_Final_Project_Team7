"""T-GCN model for multi-horizon traffic speed forecasting (Zhao et al., 2020).

Combines 2-layer GCN (spatial aggregation across the sensor graph) with
an LSTMCell (temporal modeling across input timesteps). A linear decoder
maps the final hidden state to all 12 forecast horizons at once.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


class GCNLSTMCell(nn.Module):
    """Single-timestep GCN + LSTMCell update.

    Args:
        in_channels: input features per node (2: speed + time_of_day)
        hidden_dim:  GCN output / LSTM hidden size
        num_nodes:   number of sensors in the graph
        dropout:     dropout rate between GCN layers
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        num_nodes: int,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes

        # 2-layer GCN gives a 2-hop receptive field on the road network
        self.gcn1 = GCNConv(in_channels, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # LSTMCell so we can interleave GCN and LSTM at each timestep
        self.lstm_cell = nn.LSTMCell(hidden_dim, hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        h: torch.Tensor,
        c: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Process a single timestep through GCN layers, then update LSTM state.

        Args:
            x:           (batch*num_nodes, in_channels) node features for this timestep
            edge_index:  (2, E) graph connectivity (expanded for batch)
            edge_weight: (E,) edge weights from Gaussian kernel
            h:           (batch*num_nodes, hidden_dim) LSTM hidden state from prev step
            c:           (batch*num_nodes, hidden_dim) LSTM cell state from prev step

        Returns:
            h_new, c_new: updated LSTM states, each (batch*num_nodes, hidden_dim)
        """
        out = self.gcn1(x, edge_index, edge_weight)
        out = torch.relu(out)
        out = self.dropout(out)
        out = self.gcn2(out, edge_index, edge_weight)
        out = torch.relu(out)

        h_new, c_new = self.lstm_cell(out, (h, c))
        return h_new, c_new


class GCNLSTM(nn.Module):
    """GCN+LSTM encoder-decoder for multi-horizon traffic forecasting.

    Encoder: iterates over input timesteps, running GCNLSTMCell at each step.
    Decoder: linear projection from final hidden state to all horizon steps.

    Input:  (batch, seq_len, num_nodes, in_channels)
    Output: (batch, num_nodes, out_horizon)
    """

    def __init__(
        self,
        in_channels: int = 2,
        hidden_dim: int = 64,
        out_horizon: int = 12,
        num_nodes: int = 207,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        self.out_horizon = out_horizon

        self.cell = GCNLSTMCell(in_channels, hidden_dim, num_nodes, dropout)

        self.decoder = nn.Linear(hidden_dim, out_horizon)

    def _expand_edge_index(
        self, edge_index: torch.Tensor, edge_weight: torch.Tensor, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Replicate the graph for each sample in the batch (block-diagonal structure)."""
        edge_indices = []
        edge_weights = []
        for b in range(batch_size):
            offset = b * self.num_nodes
            edge_indices.append(edge_index + offset)
            edge_weights.append(edge_weight)
        return torch.cat(edge_indices, dim=1), torch.cat(edge_weights, dim=0)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass: encode input sequence with GCN+LSTM, decode to horizon.

        Args:
            x:           (batch, seq_len, num_nodes, in_channels)
            edge_index:  (2, E) graph edge index
            edge_weight: (E,) graph edge weights

        Returns:
            (batch, num_nodes, out_horizon) predicted speeds
        """
        batch_size, seq_len, N, C = x.shape
        device = x.device

        batch_edge_index, batch_edge_weight = self._expand_edge_index(
            edge_index, edge_weight, batch_size
        )

        total_nodes = batch_size * N
        h = torch.zeros(total_nodes, self.hidden_dim, device=device)
        c = torch.zeros(total_nodes, self.hidden_dim, device=device)

        # Encode: GCN + LSTM at each timestep
        for t in range(seq_len):
            x_t = x[:, t, :, :].reshape(total_nodes, C)  # (batch*N, C)
            h, c = self.cell(x_t, batch_edge_index, batch_edge_weight, h, c)

        # Decode: project final hidden state to all horizons
        out = self.decoder(h)  # (batch*N, horizon)
        return out.reshape(batch_size, N, self.out_horizon)
