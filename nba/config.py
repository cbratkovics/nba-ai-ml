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

# Kaggle dump used for the backfill and the daily incremental ingest. There is no other
# automated source: stats.nba.com is unreachable from GitHub Actions runners (see the
# probe workflow and docs/reconciliation.md), so the dataset and model cards must describe
# this dataset as both the backfill and the daily source. They are rendered from these
# constants, never typed by hand.
KAGGLE_DATASET: str = "eoinamoore/historical-nba-data-and-player-box-scores"
KAGGLE_DATASET_TITLE: str = "Historical NBA Data and Player Box Scores"
KAGGLE_DATASET_AUTHOR: str = "Eoin Moore"
KAGGLE_DATASET_VERSION: int = 515
KAGGLE_DATASET_LICENSE: str = "CC0 1.0"
# Value written to the `source` column by the daily ingest.
KAGGLE_DAILY_SOURCE: str = "kaggle_daily"
# The daily ingest re-reads rows from this many days before the newest stored game.
DAILY_LOOKBACK_DAYS: int = 7
DAILY_REPORT_PATH: Path = Path("data") / "daily_report.json"
# Where the daily ingest puts the downloaded dump files (the slate reads the schedule here).
DUMP_DIR: Path = Path("data") / "dump"

# Published model used for scoring, pinned to a commit of HF_MODEL_REPO, and the git commit
# of the training code that produced it (reports/metrics.json `git_sha`). The model has one
# identity, shown everywhere as `commit 50a3b2e / HF fb427de` (see model_identity()).
MODEL_REVISION: str = "fb427de136e1d6c4b591ae30cf30488f44935182"
MODEL_COMMIT: str = "50a3b2e33b443d1db19274cea27467072ebfb3f8"
# Revision of HF_DATASET_REPO the published model was trained and evaluated on
# (reports/metrics.json `dataset.version`). The provenance report hashes every file at it.
DATASET_REVISION: str = "b20b5601de213fa8e704ebffaafd18f182ea68c3"
# Analyst-agent model on Groq, pinned from the catalogue queried on 2026-09-12: no Llama
# chat model was listed; openai/gpt-oss-120b was the largest model with verified native
# tool calls (0.53 s on a one-tool probe). Fallback if rate-limited: openai/gpt-oss-20b.
GROQ_MODEL: str = "openai/gpt-oss-120b"
GROQ_MODEL_FALLBACK: str = "openai/gpt-oss-20b"
# gpt-oss models reason before answering; "low" keeps a 5-7 step loop inside the 60 s wall
# clock (medium spent ~10 s per step on 2026-03-10). Valid: low, medium, high.
GROQ_REASONING_EFFORT: str = "low"

# A player is on a team's slate if they appeared in any of the team's last N games.
ROSTER_LOOKBACK_GAMES: int = 10
# Prediction and residual products (local dirs and folders in the dataset repo).
PREDICTIONS_DIR: Path = Path("predictions")
RESIDUALS_DIR: Path = Path("residuals")
HF_PREDICTIONS_PREFIX: str = "predictions"
HF_RESIDUALS_PREFIX: str = "residuals"
# Replay products: replay/<season>/{replay.json, daily_mae.json, sample_<date>.json,
# residuals/<date>.parquet}.
REPLAY_DIR: Path = Path("replay")
HF_REPLAY_PREFIX: str = "replay"
# Daily ingest reports pushed as products so later runs can read any date.
DAILY_REPORTS_DIR: Path = Path("daily_reports")
HF_DAILY_REPORTS_PREFIX: str = "daily_reports"
# Analyst-agent briefs: brief/<date>.json, brief/<date>.trace.json, latest.json, index.json.
BRIEF_DIR: Path = Path("brief")
HF_BRIEF_PREFIX: str = "brief"

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
# Kaggle API credentials (same sources). Only the daily ingest's download uses them.
KAGGLE_USERNAME: str | None = os.environ.get("KAGGLE_USERNAME") or None
KAGGLE_KEY: str | None = os.environ.get("KAGGLE_KEY") or None
# Groq API key for the analyst agent (free tier). Read here only.
GROQ_API_KEY: str | None = os.environ.get("GROQ_API_KEY") or None


def groq_api_key() -> str | None:
    return GROQ_API_KEY


def model_identity(commit: str = MODEL_COMMIT, revision: str = MODEL_REVISION) -> str:
    """The one string that names the published model: `commit <git> / HF <revision>`."""
    return f"commit {commit[:7]} / HF {revision[:7]}"


def source_lines() -> dict[str, str]:
    """Markdown bullets describing the data sources, shared by the dataset and model cards."""
    backfill = (
        f"**Historical backfill:** {KAGGLE_DATASET_AUTHOR}, *{KAGGLE_DATASET_TITLE}*, Kaggle "
        f"(`{KAGGLE_DATASET}`), version {KAGGLE_DATASET_VERSION}, {KAGGLE_DATASET_LICENSE}. "
        "Only `PlayerStatistics.csv` and `TeamHistories.csv` are used; rows carry "
        f"`source = {KAGGLE_SOURCE}`."
    )
    daily = (
        "**Daily updates:** the same Kaggle dataset, re-downloaded by the nightly GitHub "
        f"Actions job. Rows within {DAILY_LOOKBACK_DAYS} days of the newest stored game are "
        "reconciled against the stored rows; new or changed rows carry "
        f"`source = {KAGGLE_DAILY_SOURCE}`. There is no live collection from NBA.com: "
        "stats.nba.com is not reachable from GitHub Actions runners."
    )
    return {"backfill": backfill, "daily": daily}


def hf_token() -> str | None:
    """Hugging Face write token, or None when unset."""
    return HF_TOKEN
