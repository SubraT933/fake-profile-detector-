"""Configuration settings and default parameters for the fake profile detector."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATASETS_DIR = BASE_DIR / "datasets"
REPORTS_DIR = BASE_DIR / "reports"
DEFAULT_DATASET = DATASETS_DIR / "Instagram_5k.csv"

# Candidate column names representing ground truth labels across heterogeneous datasets
TARGET_COLUMNS = [
    "fake",
    "isfake",
    "is_fake",
    "labels",
    "label",
    "class",
    "target",
    "is_bot",
    "bot",
    "automatedbehaviour",
    "automatedbehavior",
    "automated_behaviour",
    "spammer",
]

# Canonical positive class indicators (1 = fake/bot, 0 = genuine/human)
POSITIVE_LABEL_VALUES = {
    "fake",
    "bot",
    "1",
    "1.0",
    "spam",
    "spammer",
    "scam",
    "true",
    "yes",
    "automated",
}

# Standard 12 features for Instagram profile evaluation
CANONICAL_FEATURES = [
    "profile pic",
    "nums/length username",
    "fullname words",
    "nums/length fullname",
    "name==username",
    "description length",
    "external URL",
    "private",
    "#posts",
    "#followers",
    "#follows",
    "followers_following_ratio",
]


@dataclass
class ModelConfig:
    """Hyperparameter and runtime configuration for model training."""

    model_type: str = "GAT"  # 'GAT' or 'GCN'
    k_neighbors: int = 8
    test_size: float = 0.20
    epochs: int = 50
    hidden_dim: int = 32
    learning_rate: float = 0.01
    weight_decay: float = 5e-4
    random_seed: int = 42
    dropout: float = 0.25
    lr_decay_factor: float = 0.5
    lr_patience: int = 5
    min_lr: float = 1e-5


@dataclass
class ExplainerConfig:
    """Configuration for local mask-based GNN explanation."""

    epochs: int = 50
    learning_rate: float = 0.04
    l1_weight: float = 0.08
    entropy_weight: float = 0.05
