"""Compatibility layer for ml_model module, delegating to fake_profile_detector."""

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fake_profile_detector.config import ModelConfig
from fake_profile_detector.data import prepare_dataset, scale_features
from fake_profile_detector.graph import construct_knn_graph
from fake_profile_detector.models.gnn import (
    GraphAttentionNetwork as GAT,
    GraphConvolutionalNetwork as GCN,
)
from fake_profile_detector.pipeline import (
    analyze_community_motif,
    explain_profile,
    predict_profile,
    train_model,
)

TARGET = "fake"


def prepare_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
    clean_df, _, _, feature_cols = prepare_dataset(df)
    return clean_df, feature_cols


def build_graph(X_scaled: np.ndarray, k: int = 6):
    adj_tensor, adj_matrix, edge_index = construct_knn_graph(X_scaled, k=k)
    return adj_tensor, adj_matrix, edge_index


def train_gnn(
    df: pd.DataFrame,
    k: int = 6,
    test_size: float = 0.2,
    epochs: int = 60,
    hidden_dim: int = 32,
    seed: int = 42,
    model_type: str = "GAT",
) -> Dict[str, Any]:
    cfg = ModelConfig(
        model_type=model_type,
        k_neighbors=k,
        test_size=test_size,
        epochs=epochs,
        hidden_dim=hidden_dim,
        random_seed=seed,
    )
    res = train_model(df, config=cfg)
    return {
        "accuracy": res["metrics"]["accuracy"],
        "precision": res["metrics"]["precision"],
        "recall": res["metrics"]["recall"],
        "f1": res["metrics"]["f1_score"],
        "confusion_matrix": np.zeros((2, 2)),
        "classification_report": {},
        "test_indices": res["test_indices"],
        "probabilities": res["probabilities"],
        "predictions": res["predictions"],
        "losses": res["losses"],
        "X_scaled": res["X_scaled"],
        "adjacency_matrix": res["adjacency_matrix"],
        "adjacency_tensor": res["adjacency_tensor"],
        "edge_index": res["edge_index"],
        "scaler": res["scaler"],
        "model": res["model"],
        "model_type": res["model_type"],
        "X_raw": res["X_raw"],
        "y": res["y"],
        "feature_cols": res["feature_names"],
        "dataframe": res["clean_df"],
        "_internal_artifact": res,
    }


def predict_user_profile(result: Dict[str, Any], profile_values: Dict[str, float], k: int = 6) -> Dict[str, Any]:
    artifact = result.get("_internal_artifact", result)
    pred = predict_profile(artifact, profile_values, k=k)
    return {
        "predicted_class": 1 if pred["is_fake"] else 0,
        "fake_probability": pred["fake_probability"],
        "neighbours": pred["nearest_indices"],
        "distances": pred["distances"],
        "weights": pred["weights"],
        "row_scaled": pred["row_scaled"],
    }


def explain_user_profile(
    result: Dict[str, Any],
    profile_values: Dict[str, float],
    k: int = 6,
    method: str = "gnn_explainer",
) -> Tuple[pd.DataFrame, np.ndarray]:
    artifact = result.get("_internal_artifact", result)
    pred = predict_profile(artifact, profile_values, k=k)
    df, edge_weights = explain_profile(artifact, pred, method=method)
    df = df.rename(columns={"feature": "Feature", "importance": "Importance"})
    return df, edge_weights


def analyze_subgraph_motif(result: Dict[str, Any], profile_values: Dict[str, float], k: int = 6) -> Dict[str, Any]:
    artifact = result.get("_internal_artifact", result)
    pred = predict_profile(artifact, profile_values, k=k)
    motif = analyze_community_motif(artifact, pred)
    return {
        "motif_type": motif["cluster_category"],
        "motif_color": "#ef4444" if motif["badge_variant"] == "danger" else ("#10b981" if motif["badge_variant"] == "success" else "#f59e0b"),
        "motif_desc": motif["description"],
        "fake_neighbor_count": motif["fake_count"],
        "real_neighbor_count": motif["genuine_count"],
        "weighted_fake_affinity": motif["weighted_affinity"],
        "neighbor_df": motif["neighbor_table"],
        "nearest": pred["nearest_indices"],
        "weights": pred["weights"],
    }
