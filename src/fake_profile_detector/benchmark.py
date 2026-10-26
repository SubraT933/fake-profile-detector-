"""Multi-dataset and multi-model benchmark evaluation engine."""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from fake_profile_detector.config import DATASETS_DIR, REPORTS_DIR, ModelConfig
from fake_profile_detector.data import load_dataset, prepare_dataset, scale_features
from fake_profile_detector.models.baselines import (
    get_baseline_models,
    train_evaluate_baseline,
)
from fake_profile_detector.pipeline import train_model

logger = logging.getLogger(__name__)


def run_benchmark_on_dataset(
    dataset_name_or_path: str,
    random_state: int = 42,
    include_gnn: bool = True,
    gnn_epochs: int = 40,
) -> pd.DataFrame:
    """Run comparative benchmark of all classifier families on a single dataset.

    Args:
        dataset_name_or_path: Filename in datasets/ or explicit file path.
        random_state: Random seed for data partitioning and estimator initialization.
        include_gnn: Whether to include GCN and GAT models in the evaluation.
        gnn_epochs: Training epoch count for GNN models.

    Returns:
        DataFrame ranking models by F1 score and balanced accuracy.
    """
    logger.info("Starting benchmark on dataset: %s", dataset_name_or_path)
    df_raw = load_dataset(dataset_name_or_path)
    clean_df, X_raw, y, feature_cols = prepare_dataset(df_raw)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw,
        y,
        test_size=0.20,
        random_state=random_state,
        stratify=y,
    )

    X_train_scaled, X_test_scaled, _ = scale_features(X_train_raw, X_test_raw)

    results: List[Dict[str, Any]] = []

    # 1. Evaluate traditional & ensemble baselines
    baselines = get_baseline_models(random_state=random_state)
    for model_name, estimator in baselines.items():
        try:
            metrics = train_evaluate_baseline(
                estimator,
                X_train_scaled,
                y_train,
                X_test_scaled,
                y_test,
            )
            metrics["model"] = model_name
            metrics["family"] = _infer_model_family(model_name)
            results.append(metrics)
        except Exception as err:
            logger.warning("Baseline %s failed: %s", model_name, err)

    # 2. Evaluate Graph Neural Networks (GCN & GAT)
    if include_gnn and len(clean_df) >= 30:
        for arch in ["GCN", "GAT"]:
            try:
                t0 = time.perf_counter()
                cfg = ModelConfig(
                    model_type=arch,
                    epochs=gnn_epochs,
                    random_seed=random_state,
                    test_size=0.20,
                )
                gnn_art = train_model(clean_df, config=cfg)
                elapsed = time.perf_counter() - t0

                gnn_metrics = gnn_art["metrics"].copy()
                gnn_metrics["model"] = f"Graph Neural Network ({arch})"
                gnn_metrics["family"] = "Graph Deep Learning"
                gnn_metrics["train_time_seconds"] = round(elapsed, 4)
                results.append(gnn_metrics)
            except Exception as err:
                logger.warning("GNN %s failed: %s", arch, err)

    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = results_df.sort_values(
            by=["f1_score", "accuracy"], ascending=[False, False]
        ).reset_index(drop=True)
        results_df.insert(0, "rank", range(1, len(results_df) + 1))

    return results_df


def run_comprehensive_benchmark(
    datasets_to_run: Optional[List[str]] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, pd.DataFrame]:
    """Execute benchmarks across multiple datasets and save persistent summaries."""
    if output_dir is None:
        output_dir = REPORTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    if datasets_to_run is None:
        datasets_to_run = [
            f.name for f in DATASETS_DIR.glob("*.csv")
            if not f.name.startswith("user_provided") and not f.name.startswith("dataset_1")
        ]

    benchmark_summaries: Dict[str, pd.DataFrame] = {}
    master_records: List[Dict[str, Any]] = []

    for ds_name in datasets_to_run:
        try:
            df_res = run_benchmark_on_dataset(ds_name)
            df_res["dataset"] = ds_name
            benchmark_summaries[ds_name] = df_res

            csv_path = output_dir / f"benchmark_{Path(ds_name).stem}.csv"
            df_res.to_csv(csv_path, index=False)
            logger.info("Saved dataset benchmark to: %s", csv_path)

            master_records.extend(df_res.to_dict(orient="records"))
        except Exception as err:
            logger.error("Failed benchmark on dataset %s: %s", ds_name, err)

    if master_records:
        master_df = pd.DataFrame(master_records)
        master_csv = output_dir / "master_benchmark_summary.csv"
        master_df.to_csv(master_csv, index=False)

        # Average performance per model across all tested datasets
        pivot = master_df.groupby("model")[
            ["accuracy", "f1_score", "precision", "recall", "roc_auc"]
        ].mean().sort_values("f1_score", ascending=False).reset_index()
        pivot.to_csv(output_dir / "model_ranking_summary.csv", index=False)

    return benchmark_summaries


def _infer_model_family(model_name: str) -> str:
    """Classify model algorithm into canonical family."""
    if any(k in model_name for k in ["Forest", "Boosting", "AdaBoost", "Ensemble", "Voting"]):
        return "Ensemble Learning"
    if "Linear" in model_name or "Logistic" in model_name or "Naive" in model_name:
        return "Linear / Probabilistic"
    if "Neighbors" in model_name:
        return "Instance-Based"
    if "Perceptron" in model_name or "MLP" in model_name:
        return "Deep Learning (MLP)"
    return "Other"
