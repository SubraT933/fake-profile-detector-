"""Compatibility layer for benchmark_engine module, delegating to fake_profile_detector."""

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fake_profile_detector.benchmark import (
    run_benchmark_on_dataset,
    run_comprehensive_benchmark,
)
from fake_profile_detector.data import prepare_dataset as prepare_any_dataset

if __name__ == "__main__":
    from run_benchmark import main
    main()
