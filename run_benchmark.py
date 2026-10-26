"""Command-line utility to run benchmarks and comparative evaluations on profile datasets."""

import argparse
import logging
import sys
from pathlib import Path

# Add src directory to path
SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fake_profile_detector.benchmark import (
    run_benchmark_on_dataset,
    run_comprehensive_benchmark,
)
from fake_profile_detector.config import DATASETS_DIR, REPORTS_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run evaluation benchmarks on social media fake profile datasets."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="Instagram_5k.csv",
        help="Target dataset name or path (default: Instagram_5k.csv).",
    )
    parser.add_argument(
        "--all-datasets",
        action="store_true",
        help="Run benchmark systematically across all available datasets.",
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="List all datasets discovered in the datasets directory.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=40,
        help="Number of epochs for GNN training (default: 40).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for data split and training (default: 42).",
    )
    parser.add_argument(
        "--no-gnn",
        action="store_true",
        help="Skip Graph Neural Network training to run tabular baselines only.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(REPORTS_DIR),
        help=f"Directory to save evaluation reports (default: {REPORTS_DIR}).",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()

    if args.list_datasets:
        datasets = sorted([f.name for f in DATASETS_DIR.glob("*.csv")])
        print("Available datasets in datasets/:")
        for ds in datasets:
            size_kb = (DATASETS_DIR / ds).stat().st_size / 1024
            print(f"  - {ds} ({size_kb:.1f} KB)")
        return

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if args.all_datasets:
        print("Executing benchmark across all available datasets...")
        summaries = run_comprehensive_benchmark(output_dir=output_path)
        print(f"\nCompleted benchmarks across {len(summaries)} datasets.")
        print(f"Reports saved to: {output_path.resolve()}")
        return

    print(f"Executing benchmark on: {args.dataset}")
    df_results = run_benchmark_on_dataset(
        dataset_name_or_path=args.dataset,
        random_state=args.seed,
        include_gnn=not args.no_gnn,
        gnn_epochs=args.epochs,
    )

    out_csv = output_path / f"benchmark_{Path(args.dataset).stem}.csv"
    df_results.to_csv(out_csv, index=False)

    print("\nBenchmark Results Leaderboard:")
    display_cols = [
        "rank",
        "model",
        "family",
        "accuracy",
        "f1_score",
        "precision",
        "recall",
        "roc_auc",
        "train_time_seconds",
    ]
    cols = [c for c in display_cols if c in df_results.columns]
    formatted = df_results[cols].to_string(index=False)
    print(formatted)
    print(f"\nReport written to: {out_csv.resolve()}")


if __name__ == "__main__":
    main()
