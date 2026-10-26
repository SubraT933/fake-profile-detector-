"""Tests for k-NN graph construction and topology properties."""

import numpy as np
import torch

from fake_profile_detector.graph import construct_knn_graph


def test_construct_knn_graph():
    np.random.seed(42)
    n_nodes = 30
    n_features = 5
    X = np.random.randn(n_nodes, n_features).astype(np.float32)

    adj_tensor, adj_matrix, edge_index = construct_knn_graph(X, k=5)

    assert adj_matrix.shape == (n_nodes, n_nodes)
    assert adj_tensor.shape == (n_nodes, n_nodes)
    assert edge_index.shape[0] == 2
    assert edge_index.shape[1] > 0

    # Test symmetry of adjacency matrix
    diff = adj_matrix - adj_matrix.T
    assert diff.nnz == 0
