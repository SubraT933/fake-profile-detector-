"""Model architectures and baseline classifiers."""

from fake_profile_detector.models.gnn import (
    GraphAttentionLayer,
    GraphAttentionNetwork,
    GraphConvolutionalNetwork,
)
from fake_profile_detector.models.baselines import (
    get_baseline_models,
    train_evaluate_baseline,
)

__all__ = [
    "GraphAttentionLayer",
    "GraphAttentionNetwork",
    "GraphConvolutionalNetwork",
    "get_baseline_models",
    "train_evaluate_baseline",
]
