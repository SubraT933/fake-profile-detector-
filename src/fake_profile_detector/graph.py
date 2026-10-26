"""Graph construction and topology utilities for feature-similarity networks."""

from typing import Tuple

import numpy as np
import scipy.sparse as sp
from scipy.sparse import diags
from sklearn.neighbors import kneighbors_graph
import torch


def construct_knn_graph(
    X: np.ndarray,
    k: int = 8,
) -> Tuple[torch.Tensor, sp.csr_matrix, torch.Tensor]:
    """Construct a symmetric k-nearest-neighbors graph from continuous feature representations.

    Args:
        X: Feature matrix of shape (num_samples, num_features).
        k: Number of nearest neighbors per node.

    Returns:
        adjacency_tensor: Normalized sparse PyTorch COO tensor for GCN propagation.
        adjacency_matrix: Unweighted symmetric sparse CSR matrix for neighbor lookups.
        edge_index: PyTorch LongTensor of shape (2, num_directed_edges) for GAT/GNNExplainer.
    """
    num_nodes = len(X)
    k_effective = max(1, min(int(k), num_nodes - 1))

    # Compute k-nearest-neighbors connectivity matrix
    knn_raw = kneighbors_graph(
        X,
        n_neighbors=k_effective,
        mode="connectivity",
        include_self=False,
    )

    # Symmetrize adjacency matrix
    symmetric_adj = knn_raw.maximum(knn_raw.T).tocsr()
    symmetric_adj.setdiag(1.0)
    symmetric_adj.eliminate_zeros()

    # Compute symmetric degree normalization: D^(-1/2) * A * D^(-1/2)
    degrees = np.asarray(symmetric_adj.sum(axis=1)).reshape(-1)
    inv_sqrt_degrees = np.power(np.maximum(degrees, 1e-12), -0.5)
    degree_matrix = diags(inv_sqrt_degrees)
    normalized_coo = (degree_matrix @ symmetric_adj @ degree_matrix).tocoo()

    # Convert normalized matrix to PyTorch sparse COO tensor
    indices = torch.tensor(
        np.vstack([normalized_coo.row, normalized_coo.col]),
        dtype=torch.long,
    )
    values = torch.tensor(normalized_coo.data, dtype=torch.float32)
    adjacency_tensor = torch.sparse_coo_tensor(
        indices,
        values,
        size=normalized_coo.shape,
    ).coalesce()

    # Create edge_index for Graph Attention Layers
    coo = symmetric_adj.tocoo()
    edge_index = torch.tensor(
        np.vstack([coo.row, coo.col]),
        dtype=torch.long,
    )

    return adjacency_tensor, symmetric_adj, edge_index
