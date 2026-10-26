"""High-level training, out-of-graph inference, and community analysis pipeline."""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F

from fake_profile_detector.config import ExplainerConfig, ModelConfig
from fake_profile_detector.data import prepare_dataset, scale_features
from fake_profile_detector.explainability import (
    compute_gradient_saliency,
    explain_subgraph_masks,
)
from fake_profile_detector.graph import construct_knn_graph
from fake_profile_detector.models.baselines import compute_classification_metrics
from fake_profile_detector.models.gnn import (
    GraphAttentionNetwork,
    GraphConvolutionalNetwork,
)

logger = logging.getLogger(__name__)


def train_model(
    df: pd.DataFrame,
    config: Optional[ModelConfig] = None,
) -> Dict[str, Any]:
    """Train a Graph Neural Network (GAT or GCN) with learning rate scheduling.

    Args:
        df: Input DataFrame containing labeled profiles.
        config: Model and training hyperparameter settings.

    Returns:
        Dictionary containing trained model, fitted scaler, graph structures, and metrics.
    """
    if config is None:
        config = ModelConfig()

    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)

    clean_df, X_raw, y, feature_cols = prepare_dataset(df)
    X_scaled, _, scaler = scale_features(X_raw)

    adj_tensor, adj_matrix, edge_index = construct_knn_graph(
        X_scaled, k=config.k_neighbors
    )

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=config.test_size,
        random_state=config.random_seed,
        stratify=y,
    )
    train_tensor = torch.tensor(train_idx, dtype=torch.long)
    test_tensor = torch.tensor(test_idx, dtype=torch.long)

    is_gat = config.model_type.upper() == "GAT"
    if is_gat:
        model: nn.Module = GraphAttentionNetwork(
            input_dim=X_scaled.shape[1],
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
        )
    else:
        model = GraphConvolutionalNetwork(
            input_dim=X_scaled.shape[1],
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
        )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.lr_decay_factor,
        patience=config.lr_patience,
        min_lr=config.min_lr,
    )

    losses: List[float] = []

    for _ in range(config.epochs):
        model.train()
        optimizer.zero_grad()

        if is_gat:
            logits = model(X_tensor, edge_index=edge_index)
        else:
            logits = model(X_tensor, adjacency=adj_tensor)

        loss = F.cross_entropy(logits[train_tensor], y_tensor[train_tensor])
        loss.backward()
        optimizer.step()

        loss_val = float(loss.detach())
        losses.append(loss_val)
        scheduler.step(loss_val)

    # Evaluation on holdout test partition
    model.eval()
    with torch.no_grad():
        if is_gat:
            logits = model(X_tensor, edge_index=edge_index)
        else:
            logits = model(X_tensor, adjacency=adj_tensor)
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()

    test_preds = probabilities[test_idx].argmax(axis=1)
    test_probs = probabilities[test_idx, 1]

    metrics = compute_classification_metrics(
        y_true=y[test_idx],
        y_pred=test_preds,
        y_prob=test_probs,
    )

    return {
        "model": model,
        "model_type": config.model_type.upper(),
        "is_gat": is_gat,
        "scaler": scaler,
        "feature_names": feature_cols,
        "X_raw": X_raw,
        "X_scaled": X_scaled,
        "y": y,
        "adjacency_matrix": adj_matrix,
        "adjacency_tensor": adj_tensor,
        "edge_index": edge_index,
        "probabilities": probabilities,
        "predictions": probabilities.argmax(axis=1),
        "test_indices": test_idx,
        "train_indices": train_idx,
        "metrics": metrics,
        "losses": losses,
        "clean_df": clean_df,
        "config": config,
    }


def predict_profile(
    artifact: Dict[str, Any],
    profile_features: Dict[str, float],
    k: Optional[int] = None,
) -> Dict[str, Any]:
    """Classify an external query profile using inverse-distance neighborhood aggregation.

    Args:
        artifact: Dictionary returned by train_model.
        profile_features: Feature dictionary matching feature_names.
        k: Optional neighbor count override.

    Returns:
        Inference results dictionary with probability, verdict, and neighbor distances.
    """
    feature_names = artifact["feature_names"]
    input_vector = [profile_features.get(f, 0.0) for f in feature_names]
    row_df = pd.DataFrame([input_vector], columns=feature_names)

    row_scaled = artifact["scaler"].transform(row_df.to_numpy()).astype(np.float32)[0]
    X_scaled = artifact["X_scaled"]

    if k is None:
        k = artifact["config"].k_neighbors
    k = max(1, min(int(k), len(X_scaled)))

    distances = np.linalg.norm(X_scaled - row_scaled, axis=1)
    nearest_indices = np.argsort(distances)[:k]

    # Compute inverse distance weights
    inv_dist = 1.0 / (distances[nearest_indices] + 1e-6)
    weights = inv_dist / np.sum(inv_dist)

    neighbor_fake_probs = artifact["probabilities"][nearest_indices, 1]
    fake_probability = float(np.sum(neighbor_fake_probs * weights))
    is_fake = bool(fake_probability >= 0.50)

    return {
        "is_fake": is_fake,
        "fake_probability": fake_probability,
        "confidence": fake_probability if is_fake else (1.0 - fake_probability),
        "row_scaled": row_scaled,
        "nearest_indices": nearest_indices,
        "distances": distances[nearest_indices],
        "weights": weights,
    }


def analyze_community_motif(
    artifact: Dict[str, Any],
    prediction_result: Dict[str, Any],
) -> Dict[str, Any]:
    """Analyze the topological structure of the profile's local neighborhood."""
    nearest = prediction_result["nearest_indices"]
    distances = prediction_result["distances"]
    weights = prediction_result["weights"]
    labels = artifact["y"][nearest]

    fake_count = int(np.sum(labels == 1))
    genuine_count = int(np.sum(labels == 0))
    weighted_affinity = float(np.sum((labels == 1) * weights))

    if weighted_affinity >= 0.65:
        category = "Coordinated Synthetic Cluster"
        description = (
            "Profile clusters tightly with synthetic or automated accounts. "
            "Behavioral characteristics strongly resemble coordinated bot farm patterns."
        )
        badge_variant = "danger"
    elif weighted_affinity <= 0.35:
        category = "Organic Human Cluster"
        description = (
            "Profile embeds within verified authentic user topology, displaying typical "
            "follower-to-following behavior and organic activity indicators."
        )
        badge_variant = "success"
    else:
        category = "Mixed / Boundary Topology"
        description = (
            "Profile resides along the decision boundary between genuine and synthetic clusters, "
            "sharing traits with both populations."
        )
        badge_variant = "warning"

    neighbor_df = artifact["clean_df"].iloc[nearest].copy()
    neighbor_df.insert(0, "neighbor_index", nearest)
    neighbor_df["distance"] = np.round(distances, 3)
    neighbor_df["influence_weight"] = np.round(weights, 4)
    neighbor_df["status"] = ["Synthetic" if lbl == 1 else "Genuine" for lbl in labels]

    return {
        "cluster_category": category,
        "description": description,
        "badge_variant": badge_variant,
        "fake_count": fake_count,
        "genuine_count": genuine_count,
        "weighted_affinity": weighted_affinity,
        "neighbor_table": neighbor_df,
    }


def explain_profile(
    artifact: Dict[str, Any],
    prediction_result: Dict[str, Any],
    method: str = "gnn_explainer",
    config: Optional[ExplainerConfig] = None,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Generate feature attribution and edge influence scores for a query profile."""
    if method == "gnn_explainer":
        query_scaled = prediction_result["row_scaled"]
        neighbors_scaled = artifact["X_scaled"][prediction_result["nearest_indices"]]
        return explain_subgraph_masks(
            model=artifact["model"],
            query_vector_scaled=query_scaled,
            neighbor_vectors_scaled=neighbors_scaled,
            feature_names=artifact["feature_names"],
            is_gat=artifact["is_gat"],
            config=config,
        )

    # Fallback to gradient saliency
    first_idx = prediction_result["nearest_indices"][0]
    df = compute_gradient_saliency(
        model=artifact["model"],
        X_scaled=artifact["X_scaled"],
        target_idx=int(first_idx),
        feature_names=artifact["feature_names"],
        edge_index=artifact["edge_index"],
        adjacency_tensor=artifact["adjacency_tensor"],
        is_gat=artifact["is_gat"],
    )
    edge_weights = prediction_result["weights"]
    return df, edge_weights
