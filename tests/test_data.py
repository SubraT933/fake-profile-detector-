"""Tests for dataset loading, preparation, and feature scaling."""

import numpy as np
import pandas as pd
import pytest

from fake_profile_detector.data import (
    detect_target_column,
    prepare_dataset,
    scale_features,
)


def test_detect_target_column():
    df1 = pd.DataFrame({"feature1": [1, 2], "fake": [0, 1]})
    assert detect_target_column(df1) == "fake"

    df2 = pd.DataFrame({"feature1": [1, 2], "is_bot": [0, 1]})
    assert detect_target_column(df2) == "is_bot"

    df3 = pd.DataFrame({"feature1": [1, 2], "unknown_col": [0, 1]})
    with pytest.raises(ValueError):
        detect_target_column(df3)


def test_prepare_dataset():
    np.random.seed(42)
    n_samples = 40
    data = {
        "profile pic": np.random.choice([0.0, 1.0], size=n_samples),
        "#followers": np.random.randint(10, 5000, size=n_samples),
        "#follows": np.random.randint(10, 2000, size=n_samples),
        "nums/length username": np.random.uniform(0.0, 0.5, size=n_samples),
        "fake": np.random.choice([0, 1], size=n_samples),
    }
    raw_df = pd.DataFrame(data)

    clean_df, X, y, features = prepare_dataset(raw_df)

    assert len(clean_df) == n_samples
    assert X.shape[0] == n_samples
    assert len(y) == n_samples
    assert set(np.unique(y)).issubset({0, 1})
    assert len(features) >= 4


def test_scale_features():
    X = np.array([[1.0, 100.0], [2.0, 200.0], [3.0, 300.0]], dtype=np.float32)
    X_train_scaled, _, scaler = scale_features(X)

    assert X_train_scaled.shape == X.shape
    assert np.allclose(X_train_scaled.mean(axis=0), 0.0, atol=1e-5)
