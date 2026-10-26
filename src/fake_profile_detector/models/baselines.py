"""Standard machine learning baselines and ensemble classifiers."""

import time
from typing import Any, Dict

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    AdaBoostClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC


def get_baseline_models(random_state: int = 42) -> Dict[str, Any]:
    """Instantiate a suite of competitive baseline classifiers.

    Args:
        random_state: Random seed for reproducible training.

    Returns:
        Dictionary mapping model names to scikit-learn estimators.
    """
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=8, random_state=random_state, n_jobs=-1
    )
    gbdt = HistGradientBoostingClassifier(
        max_iter=100, max_depth=6, random_state=random_state
    )
    ada = AdaBoostClassifier(
        n_estimators=50, random_state=random_state
    )
    lr = LogisticRegression(
        max_iter=1000, C=1.0, random_state=random_state
    )
    svc = CalibratedClassifierCV(
        LinearSVC(C=1.0, max_iter=2000, random_state=random_state)
    )
    knn = KNeighborsClassifier(
        n_neighbors=7, n_jobs=-1
    )
    nb = GaussianNB()
    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        max_iter=300,
        early_stopping=True,
        random_state=random_state,
    )
    voting = VotingClassifier(
        estimators=[("rf", rf), ("gbdt", gbdt), ("lr", lr)],
        voting="soft",
    )

    return {
        "Random Forest": rf,
        "HistGradientBoosting": gbdt,
        "AdaBoost": ada,
        "Logistic Regression": lr,
        "Linear SVM (Calibrated)": svc,
        "K-Nearest Neighbors": knn,
        "Naive Bayes": nb,
        "Multi-Layer Perceptron (MLP)": mlp,
        "Voting Ensemble (RF+GBDT+LR)": voting,
    }


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, float]:
    """Compute rigorous classification metrics.

    Args:
        y_true: Ground truth binary targets.
        y_pred: Predicted discrete classes.
        y_prob: Predicted positive class probabilities.

    Returns:
        Dictionary containing standard evaluation metrics.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        roc_auc = 0.5

    try:
        pr_auc = float(average_precision_score(y_true, y_prob))
    except Exception:
        pr_auc = float(np.mean(y_true))

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }


def train_evaluate_baseline(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, Any]:
    """Fit a scikit-learn estimator and evaluate performance metrics."""
    start_time = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - start_time

    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        decision = model.decision_function(X_test)
        y_prob = 1.0 / (1.0 + np.exp(-decision))
    else:
        y_prob = y_pred.astype(float)

    metrics = compute_classification_metrics(y_test, y_pred, y_prob)
    metrics["train_time_seconds"] = round(train_time, 4)
    return metrics
