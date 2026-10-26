"""Tests for training, inference, and community motif detection pipeline."""

import numpy as np
import pandas as pd

from fake_profile_detector.config import ModelConfig
from fake_profile_detector.pipeline import (
    analyze_community_motif,
    predict_profile,
    train_model,
)


def test_pipeline_train_and_predict():
    np.random.seed(42)
    n = 60
    df = pd.DataFrame({
        "profile pic": np.random.choice([0.0, 1.0], size=n),
        "nums/length username": np.random.uniform(0.0, 0.5, size=n),
        "fullname words": np.random.choice([1.0, 2.0, 3.0], size=n),
        "nums/length fullname": np.zeros(n),
        "name==username": np.zeros(n),
        "description length": np.random.uniform(0.0, 80.0, size=n),
        "external URL": np.random.choice([0.0, 1.0], size=n),
        "private": np.random.choice([0.0, 1.0], size=n),
        "#posts": np.random.randint(1, 500, size=n).astype(float),
        "#followers": np.random.randint(5, 5000, size=n).astype(float),
        "#follows": np.random.randint(5, 2000, size=n).astype(float),
        "followers_following_ratio": np.random.uniform(0.1, 10.0, size=n),
        "fake": np.random.choice([0, 1], size=n),
    })

    config = ModelConfig(epochs=5, k_neighbors=4, hidden_dim=16)
    artifact = train_model(df, config=config)

    assert "model" in artifact
    assert "metrics" in artifact
    assert artifact["metrics"]["accuracy"] >= 0.0

    # Test out-of-graph inference
    query_features = {col: 1.0 for col in artifact["feature_names"]}
    pred = predict_profile(artifact, query_features, k=4)

    assert "is_fake" in pred
    assert "fake_probability" in pred
    assert 0.0 <= pred["fake_probability"] <= 1.0
    assert len(pred["nearest_indices"]) == 4

    # Test community motif analysis
    motif = analyze_community_motif(artifact, pred)
    assert "cluster_category" in motif
    assert "weighted_affinity" in motif
    assert len(motif["neighbor_table"]) == 4
