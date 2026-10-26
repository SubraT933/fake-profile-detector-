"""Local subgraph explainability and feature attribution engines."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from fake_profile_detector.config import ExplainerConfig


def compute_gradient_saliency(
    model: nn.Module,
    X_scaled: np.ndarray,
    target_idx: int,
    feature_names: List[str],
    edge_index: Optional[torch.Tensor] = None,
    adjacency_tensor: Optional[torch.Tensor] = None,
    is_gat: bool = True,
) -> pd.DataFrame:
    """Compute gradient-times-input saliency for a given node.

    Args:
        model: Trained PyTorch GNN model.
        X_scaled: Scaled feature matrix of the graph.
        target_idx: Index of node to explain.
        feature_names: List of feature column names.
        edge_index: Graph edge indices for GAT.
        adjacency_tensor: Sparse normalized adjacency tensor for GCN.
        is_gat: True if GAT, False if GCN.

    Returns:
        DataFrame containing sorted feature attribution scores.
    """
    model.eval()
    model.zero_grad()

    x = torch.tensor(X_scaled, dtype=torch.float32, requires_grad=True)
    if is_gat:
        logits = model(x, edge_index=edge_index)
    else:
        logits = model(x, adjacency=adjacency_tensor)

    pred_class = int(logits[target_idx].argmax().item())
    logits[target_idx, pred_class].backward()

    grads = x.grad[target_idx].detach().cpu().numpy()
    attributions = np.abs(grads * X_scaled[target_idx])

    df = pd.DataFrame({
        "feature": feature_names,
        "importance": attributions,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    return df


def explain_subgraph_masks(
    model: nn.Module,
    query_vector_scaled: np.ndarray,
    neighbor_vectors_scaled: np.ndarray,
    feature_names: List[str],
    is_gat: bool = True,
    config: Optional[ExplainerConfig] = None,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Optimize feature and edge masks for local subgraph explainability (GNNExplainer).

    Args:
        model: Trained PyTorch GNN model.
        query_vector_scaled: Scaled feature vector of the evaluated query profile.
        neighbor_vectors_scaled: Scaled feature vectors of top-k dataset neighbors.
        feature_names: List of feature names.
        is_gat: True if GAT model, False if GCN.
        config: Explainer hyperparameter configuration.

    Returns:
        Tuple of (feature_importance_df, edge_weights_array).
    """
    if config is None:
        config = ExplainerConfig()

    num_features = len(feature_names)
    k_neighbors = len(neighbor_vectors_scaled)

    # Local subgraph: query node is 0, dataset neighbors are 1..k
    subgraph_features = np.vstack([query_vector_scaled, neighbor_vectors_scaled])
    sub_x = torch.tensor(subgraph_features, dtype=torch.float32)
    num_nodes = sub_x.shape[0]

    # Star-graph connectivity between query node and each neighbor
    src = [0] * k_neighbors + list(range(1, k_neighbors + 1))
    dst = list(range(1, k_neighbors + 1)) + [0] * k_neighbors
    sub_edge_index = torch.tensor([src, dst], dtype=torch.long)

    # Trainable masks
    feat_param = nn.Parameter(torch.zeros(num_features))
    edge_param = nn.Parameter(torch.zeros(len(src)))

    optimizer = torch.optim.Adam([feat_param, edge_param], lr=config.learning_rate)
    model.eval()

    # Determine baseline class label
    with torch.no_grad():
        if is_gat:
            base_logits = model(sub_x, edge_index=sub_edge_index)
        else:
            adj_dense = torch.zeros((num_nodes, num_nodes))
            adj_dense[sub_edge_index[0], sub_edge_index[1]] = 1.0
            for i in range(num_nodes):
                adj_dense[i, i] = 1.0
            base_logits = model(sub_x, adjacency=adj_dense.to_sparse())
        target_class = int(base_logits[0].argmax().item())

    # Optimization loop
    for _ in range(config.epochs):
        optimizer.zero_grad()
        m_feat = torch.sigmoid(feat_param)
        m_edge = torch.sigmoid(edge_param)

        masked_x = sub_x * m_feat.unsqueeze(0)

        if is_gat:
            out_logits = model(
                masked_x,
                edge_index=sub_edge_index,
                edge_weight_mask=m_edge,
            )
        else:
            adj_dense = torch.zeros((num_nodes, num_nodes))
            adj_dense[sub_edge_index[0], sub_edge_index[1]] = m_edge
            for i in range(num_nodes):
                adj_dense[i, i] = 1.0
            out_logits = model(masked_x, adjacency=adj_dense.to_sparse())

        loss_pred = F.cross_entropy(
            out_logits[0:1],
            torch.tensor([target_class], device=sub_x.device),
        )

        # Sparsity and entropy penalties
        loss_sparsity = torch.mean(m_feat) + torch.mean(m_edge)
        entropy_feat = -torch.mean(
            m_feat * torch.log(m_feat + 1e-12) + (1.0 - m_feat) * torch.log(1.0 - m_feat + 1e-12)
        )
        entropy_edge = -torch.mean(
            m_edge * torch.log(m_edge + 1e-12) + (1.0 - m_edge) * torch.log(1.0 - m_edge + 1e-12)
        )

        loss = (
            loss_pred
            + config.l1_weight * loss_sparsity
            + config.entropy_weight * (entropy_feat + entropy_edge)
        )
        loss.backward()
        optimizer.step()

    final_feat_mask = torch.sigmoid(feat_param).detach().cpu().numpy()
    final_edge_mask = torch.sigmoid(edge_param).detach().cpu().numpy()[:k_neighbors]

    feature_df = pd.DataFrame({
        "feature": feature_names,
        "importance": final_feat_mask,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    return feature_df, final_edge_mask
