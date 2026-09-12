"""Pipeline settings in one place.

This is the only module in the package that reads the process environment.
Everything else imports the constants and helpers defined here.
"""

from __future__ import annotations

import os
from pathlib import Path

# Seasons in the dataset. The last one is held out for evaluation.
SEASONS: tuple[str, ...] = ("2021-22", "2022-23", "2023-24", "2024-25", "2025-26")
HOLDOUT_SEASON: str = "2025-26"
TRAIN_SEASONS: tuple[str, ...] = tuple(s for s in SEASONS if s != HOLDOUT_SEASON)

# Only these game types are kept during backfill (Kaggle `gameType` values).
GAME_TYPES: tuple[str, ...] = ("Regular Season",)

# Feature settings.
ROLLING_WINDOWS: tuple[int, ...] = (5, 10, 20)

# Rows where the player logged fewer minutes than this are excluded from
# training and evaluation. They still count as history for feature building.
MIN_MINUTES: float = 10.0

TARGETS: tuple[str, ...] = ("pts", "reb", "ast")

# Hugging Face repositories.
HF_DATASET_REPO: str = "cbratkovics/nba-game-logs"
HF_MODEL_REPO: str = "cbratkovics/nba-stat-predictor"
# Folder inside the dataset repo that holds the per-season Parquet files.
HF_DATASET_PREFIX: str = "game_logs"

# Local paths (relative to the repo root).
DATA_DIR: Path = Path("data") / "game_logs"
MODELS_DIR: Path = Path("models")
REPORTS_DIR: Path = Path("reports")
METRICS_PATH: Path = REPORTS_DIR / "metrics.json"

# Value written to the `source` column by the Kaggle backfill.
KAGGLE_SOURCE: str = "kaggle_v515"

RANDOM_SEED: int = 42

# One LightGBM regressor per target, same settings for each.
LGBM_PARAMS: dict[str, object] = {
    "n_estimators": 600,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_child_samples": 50,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "random_state": RANDOM_SEED,
    "verbose": -1,
}


def hf_token() -> str | None:
    """Hugging Face write token, or None when unset."""
    return os.environ.get("HF_TOKEN")
