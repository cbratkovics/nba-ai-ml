"""Pipeline settings in one place.

This is the only module in the package that reads the process environment.
Everything else imports the constants and helpers defined here.

A `.env` file at the repository root is loaded first (it is git-ignored).
Variables already present in the environment win over values in the file.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env", override=False)

# Seasons in the dataset. The last one is held out for evaluation.
SEASONS: tuple[str, ...] = ("2021-22", "2022-23", "2023-24", "2024-25", "2025-26")
HOLDOUT_SEASON: str = "2025-26"
TRAIN_SEASONS: tuple[str, ...] = tuple(s for s in SEASONS if s != HOLDOUT_SEASON)

# Kaggle `gameType` values kept during backfill. NBA Cup (in-season tournament)
# group and knockout games count as regular-season games in official accounting;
# the dump labels them inconsistently by season, so every Cup label is kept and the
# Cup final is excluded separately in nba.ingest.kaggle_backfill.
GAME_TYPES: tuple[str, ...] = (
    "Regular Season",
    "NBA Emirates Cup",
    "Emirates NBA Cup",
    "NBA Cup",
    "in-season-knockout",
)

# Feature settings.
ROLLING_WINDOWS: tuple[int, ...] = (5, 10, 20)

# Rows where the player logged fewer minutes than this are excluded from
# training and evaluation. They still count as history for feature building.
MIN_MINUTES: float = 10.0

TARGETS: tuple[str, ...] = ("pts", "reb", "ast")

# Backfill sanity thresholds per season. Below the minimums the backfill stops;
# a season with fewer than FULL_SEASON_GAMES games only warns (known upstream gaps).
MIN_ROWS_PER_SEASON: int = 20_000
MIN_GAMES_PER_SEASON: int = 1_200
FULL_SEASON_GAMES: int = 1_230  # 30 teams x 82 games / 2

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


# Hugging Face write token from the environment or the repo-root .env file.
HF_TOKEN: str | None = os.environ.get("HF_TOKEN") or None


def hf_token() -> str | None:
    """Hugging Face write token, or None when unset."""
    return HF_TOKEN
