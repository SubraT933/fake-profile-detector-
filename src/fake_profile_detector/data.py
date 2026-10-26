"""Dataset loading, schema harmonization, cleaning, and preprocessing."""

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from fake_profile_detector.config import (
    DATASETS_DIR,
    POSITIVE_LABEL_VALUES,
    TARGET_COLUMNS,
)

logger = logging.getLogger(__name__)

EXCEL_ERROR_VALUES = ["#DIV/0!", "#VALUE!", "#REF!", "#NUM!", "#N/A", "null", "none", "nan"]

METADATA_COLUMNS_TO_DROP = {
    "id",
    "user_id",
    "id_str",
    "screen_name",
    "username",
    "name",
    "description",
    "bio",
    "created_at",
    "url",
    "lang",
    "location",
    "profile_image_url",
    "avatar_url",
    "extracted_at",
    "unnamed: 0",
}


def load_dataset(source: Union[str, Path]) -> pd.DataFrame:
    """Load a tabular dataset from a path or dataset name in the datasets directory.

    Args:
        source: File path or base filename.

    Returns:
        Loaded pandas DataFrame.
    """
    path = Path(source)
    if not path.is_file():
        candidate = DATASETS_DIR / source
        if candidate.is_file():
            path = candidate
        elif candidate.with_suffix(".csv").is_file():
            path = candidate.with_suffix(".csv")
        else:
            raise FileNotFoundError(f"Dataset not found at '{source}' or '{candidate}'")

    logger.info("Loading dataset from: %s", path)
    df = pd.read_csv(path)
    return df


def detect_target_column(df: pd.DataFrame) -> str:
    """Identify the label/target column from candidates."""
    normalized_cols = {
        col.lower().strip().replace(" ", "").replace("_", ""): col
        for col in df.columns
    }

    for candidate in TARGET_COLUMNS:
        normalized_cand = candidate.replace("_", "")
        if normalized_cand in normalized_cols:
            return normalized_cols[normalized_cand]

    raise ValueError(
        f"Target column not found. Available columns: {list(df.columns)}"
    )


def prepare_dataset(
    df: pd.DataFrame,
    target_column: Optional[str] = None,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[str]]:
    """Clean and preprocess tabular dataset with automatic target and feature detection.

    Args:
        df: Input raw DataFrame.
        target_column: Explicit target column name, or None to auto-detect.

    Returns:
        Tuple of (clean_df, X, y, feature_cols).
    """
    cleaned = df.copy()
    cleaned = cleaned.replace(EXCEL_ERROR_VALUES, np.nan)
    cleaned = cleaned.replace([np.inf, -np.inf], np.nan)

    if target_column is None:
        target_column = detect_target_column(cleaned)

    # Encode target column as binary integer (1 = fake/bot, 0 = genuine/human)
    target_series = cleaned[target_column].astype(str).str.strip().str.lower()
    y = target_series.apply(lambda val: 1 if val in POSITIVE_LABEL_VALUES else 0).to_numpy(dtype=np.int64)

    # Identify candidate feature columns
    cols_to_exclude = {target_column.lower()} | METADATA_COLUMNS_TO_DROP
    potential_features = [
        col for col in cleaned.columns
        if col.lower().strip() not in cols_to_exclude
    ]

    # Convert features to numeric, dropping columns that cannot be cast
    numeric_features: List[str] = []
    for col in potential_features:
        converted = pd.to_numeric(cleaned[col], errors="coerce")
        # Keep column if at least 25% of entries are valid numeric values
        if converted.notna().sum() >= max(int(0.25 * len(cleaned)), 5):
            cleaned[col] = converted
            numeric_features.append(col)
        else:
            cleaned.drop(columns=[col], inplace=True, errors="ignore")

    if not numeric_features:
        raise ValueError("No valid numeric feature columns found in dataset.")

    # Impute missing values with median, fallback to 0
    medians = cleaned[numeric_features].median(numeric_only=True)
    cleaned[numeric_features] = cleaned[numeric_features].fillna(medians).fillna(0.0)

    # Compute follower-to-following ratio if posts/followers exist and ratio is missing
    ratio_col_candidates = [c for c in numeric_features if "ratio" in c.lower()]
    if not ratio_col_candidates:
        followers_cols = [c for c in numeric_features if "follower" in c.lower() or "followee" in c.lower()]
        following_cols = [c for c in numeric_features if "following" in c.lower() or "follows" in c.lower() or "friends" in c.lower()]
        if followers_cols and following_cols:
            f_col = followers_cols[0]
            ing_col = following_cols[0]
            ratio_name = "followers_following_ratio"
            cleaned[ratio_name] = cleaned[f_col] / np.maximum(cleaned[ing_col], 1.0)
            numeric_features.append(ratio_name)

    X = cleaned[numeric_features].to_numpy(dtype=np.float32)
    cleaned["target"] = y

    if len(np.unique(y)) < 2:
        raise ValueError("Dataset must contain at least two classes (both positive and negative instances).")

    if len(cleaned) < 20:
        raise ValueError(f"Dataset has only {len(cleaned)} rows. Minimum requirement is 20 rows.")

    return cleaned, X, y, numeric_features


def scale_features(
    X_train: np.ndarray,
    X_test: Optional[np.ndarray] = None,
    scaler: Optional[StandardScaler] = None,
    clip_val: float = 10.0,
) -> Tuple[np.ndarray, Optional[np.ndarray], StandardScaler]:
    """Fit or apply standard scaling to feature matrices with outlier clipping.

    Args:
        X_train: Training feature matrix.
        X_test: Optional evaluation feature matrix.
        scaler: Existing fitted scaler, or None to fit a new one.
        clip_val: Symmetric bound to clip scaled values.

    Returns:
        Tuple of (X_train_scaled, X_test_scaled, fitted_scaler).
    """
    if scaler is None:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
    else:
        X_train_scaled = scaler.transform(X_train)

    X_train_scaled = np.clip(X_train_scaled, -clip_val, clip_val).astype(np.float32)

    X_test_scaled = None
    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
        X_test_scaled = np.clip(X_test_scaled, -clip_val, clip_val).astype(np.float32)

    return X_train_scaled, X_test_scaled, scaler
