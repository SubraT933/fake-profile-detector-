"""Compatibility layer for url_extractor module, delegating to fake_profile_detector."""

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fake_profile_detector.extractor import (
    SAMPLE_PROFILES,
    extract_profile_features as extract_features_from_instagram_url,
    parse_numeric_shorthand as parse_count,
    sanitize_handle as clean_instagram_input,
)

__all__ = [
    "SAMPLE_PROFILES",
    "clean_instagram_input",
    "extract_features_from_instagram_url",
    "parse_count",
]
