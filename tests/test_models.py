"""Tests for GNN and baseline models."""

import numpy as np
import torch

from fake_profile_detector.models.baselines import (
    compute_classification_metrics,
    get_baseline_models,
)
from fake_profile_detector.models.gnn import (
    GraphAttentionNetwork,
    GraphConvolutionalNetwork,
)


def test_gcn_forward():
    num_nodes = 20
    feat_dim = 6
    x = torch.randn(num_nodes, feat_dim)
    adj_dense = torch.eye(num_nodes)
    adj_sparse = adj_dense.to_sparse()

    gcn = GraphConvolutionalNetwork(input_dim=feat_dim, hidden_dim=16)
    out = gcn(x, adjacency=adj_sparse)

    assert out.shape == (num_nodes, 2)


def test_gat_forward():
    num_nodes = 20
    feat_dim = 6
    x = torch.randn(num_nodes, feat_dim)
    src = torch.arange(num_nodes)
    dst = (src + 1) % num_nodes
    edge_index = torch.stack([src, dst], dim=0)

    gat = GraphAttentionNetwork(input_dim=feat_dim, hidden_dim=16)
    out = gat(x, edge_index=edge_index)

    assert out.shape == (num_nodes, 2)


def test_compute_classification_metrics():
    y_true = np.array([0, 1, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    y_prob = np.array([0.1, 0.9, 0.2, 0.4, 0.8])

    metrics = compute_classification_metrics(y_true, y_pred, y_prob)

    assert "accuracy" in metrics
    assert "f1_score" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0
