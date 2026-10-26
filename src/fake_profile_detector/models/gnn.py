"""Graph Neural Network architectures: GCN and Graph Attention Networks (GAT)."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolutionalNetwork(nn.Module):
    """Two-layer Graph Convolutional Network (Kipf & Welling)."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        num_classes: int = 2,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, num_classes)
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        edge_weight_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Perform message passing using normalized sparse adjacency tensor."""
        h = torch.sparse.mm(adjacency, x)
        h = F.relu(self.layer1(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = torch.sparse.mm(adjacency, h)
        return self.layer2(h)


class GraphAttentionLayer(nn.Module):
    """Single-head Graph Attention Layer with parameterized edge attention."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout: float = 0.20,
        alpha: float = 0.20,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha

        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.attention_vector = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.linear.weight.data, gain=1.414)
        nn.init.xavier_uniform_(self.attention_vector.data, gain=1.414)
        self.leaky_relu = nn.LeakyReLU(self.alpha)

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: int,
        edge_weight_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute attention-weighted neighborhood aggregation."""
        wh = self.linear(h)
        row, col = edge_index[0], edge_index[1]

        # Concatenate source and destination projected node features
        edge_repr = torch.cat([wh[row], wh[col]], dim=1)
        raw_attention = self.leaky_relu(
            torch.matmul(edge_repr, self.attention_vector).squeeze(1)
        )

        # Numerically stable attention softmax over neighborhood
        max_per_node = torch.zeros(num_nodes, device=h.device).scatter_reduce(
            0, row, raw_attention, reduce="amax", include_self=False
        )
        attention_exp = torch.exp(raw_attention - max_per_node[row])

        if edge_weight_mask is not None:
            attention_exp = attention_exp * edge_weight_mask

        sum_per_node = torch.zeros(num_nodes, device=h.device).scatter_add_(
            0, row, attention_exp
        )
        coefficients = attention_exp / (sum_per_node[row] + 1e-12)
        coefficients_dropped = F.dropout(
            coefficients, p=self.dropout, training=self.training
        )

        # Aggregate weighted node messages
        weighted_features = wh[col] * coefficients_dropped.unsqueeze(1)
        output = torch.zeros(
            num_nodes, self.out_features, device=h.device
        ).scatter_add_(
            0,
            row.unsqueeze(1).expand(-1, self.out_features),
            weighted_features,
        )

        return output, coefficients


class GraphAttentionNetwork(nn.Module):
    """Two-layer Graph Attention Network (GAT) for node classification."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        num_classes: int = 2,
        dropout: float = 0.20,
    ) -> None:
        super().__init__()
        self.gat1 = GraphAttentionLayer(input_dim, hidden_dim, dropout=dropout)
        self.gat2 = GraphAttentionLayer(hidden_dim, num_classes, dropout=dropout)
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        adjacency: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None,
        edge_weight_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> torch.Tensor:
        """Forward pass through GAT layers."""
        if edge_index is None:
            raise ValueError("edge_index must be provided for GraphAttentionNetwork.")

        num_nodes = x.size(0)
        h, attn1 = self.gat1(
            x, edge_index, num_nodes, edge_weight_mask=edge_weight_mask
        )
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        out, attn2 = self.gat2(
            h, edge_index, num_nodes, edge_weight_mask=edge_weight_mask
        )

        if return_attention:
            return out, (attn1, attn2)
        return out
