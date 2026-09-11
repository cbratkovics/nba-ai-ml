"""Train one LightGBM regressor per target and score it against two baselines.

Split: train on config.TRAIN_SEASONS, hold out all of config.HOLDOUT_SEASON.
Baselines: the player's mean over the previous 10 games, and the player's
season-to-date mean. Both are already features, so they are read from the
feature frame directly.

Usage:
    python -m nba.models.train [--data-dir data/game_logs] [--models-dir models]
                               [--metrics-path reports/metrics.json]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from nba import config
from nba.features import asof
from nba.models import evaluate
from nba.storage import local

BASELINE_WINDOW = 10
MODEL_FILE_TEMPLATE = "lgbm_{target}.txt"


def model_path(models_dir: Path, target: str) -> Path:
    return models_dir / MODEL_FILE_TEMPLATE.format(target=target)


def score(y_true: pd.Series, y_pred: np.ndarray | pd.Series) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "r2": float(r2_score(y_true, y_pred)),
        "n": int(len(y_true)),
    }


def split(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    played = features["minutes"] >= config.MIN_MINUTES
    train = features[played & features["season"].isin(config.TRAIN_SEASONS)]
    holdout = features[played & (features["season"] == config.HOLDOUT_SEASON)]
    return train, holdout


def baseline_columns(target: str) -> dict[str, str]:
    return {
        f"baseline_last{BASELINE_WINDOW}": f"{target}_mean_last{BASELINE_WINDOW}",
        "baseline_season": f"{target}_mean_season",
    }


def evaluation_rows(holdout: pd.DataFrame, target: str) -> pd.DataFrame:
    """Holdout rows where both baselines are defined, so all three are scored on the same rows."""
    cols = list(baseline_columns(target).values())
    return holdout[holdout[cols].notna().all(axis=1)]


def fit(
    train: pd.DataFrame, target: str, params: dict[str, Any] | None = None
) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(**(params or config.LGBM_PARAMS))
    model.fit(train[asof.FEATURE_COLUMNS], train[target])
    return model


def evaluate_target(
    model: lgb.LGBMRegressor, holdout: pd.DataFrame, target: str
) -> dict[str, dict[str, float]]:
    rows = evaluation_rows(holdout, target)
    y = rows[target]
    results = {"model": score(y, model.predict(rows[asof.FEATURE_COLUMNS]))}
    for name, col in baseline_columns(target).items():
        results[name] = score(y, rows[col])
    return results


def _date_range(df: pd.DataFrame) -> dict[str, str | None]:
    if df.empty:
        return {"start": None, "end": None}
    return {
        "start": df["game_date"].min().date().isoformat(),
        "end": df["game_date"].max().date().isoformat(),
    }


def run(
    game_logs: pd.DataFrame,
    models_dir: Path,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build features, fit one model per target, evaluate, save models. Returns a results dict."""
    if BASELINE_WINDOW not in config.ROLLING_WINDOWS:
        raise ValueError(
            f"baseline window {BASELINE_WINDOW} must be one of {config.ROLLING_WINDOWS}"
        )
    params = params or config.LGBM_PARAMS
    features = asof.build_features(game_logs)
    train, holdout = split(features)
    if train.empty or holdout.empty:
        raise ValueError(
            f"empty split: {len(train)} train rows, {len(holdout)} holdout rows "
            f"(train seasons {config.TRAIN_SEASONS}, holdout {config.HOLDOUT_SEASON})"
        )

    models_dir.mkdir(parents=True, exist_ok=True)
    targets: dict[str, Any] = {}
    for target in config.TARGETS:
        model = fit(train, target, params)
        model.booster_.save_model(str(model_path(models_dir, target)))
        targets[target] = evaluate_target(model, holdout, target)

    return {
        "targets": targets,
        "split": {
            "train_seasons": list(config.TRAIN_SEASONS),
            "holdout_season": config.HOLDOUT_SEASON,
            "train_dates": _date_range(train),
            "holdout_dates": _date_range(holdout),
            "n_train_rows": int(len(train)),
            "n_holdout_rows": int(len(holdout)),
            "min_minutes": config.MIN_MINUTES,
        },
        "features": list(asof.FEATURE_COLUMNS),
        "lgbm_params": {k: v for k, v in params.items() if k != "verbose"},
        "model_files": {t: model_path(models_dir, t).name for t in config.TARGETS},
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    parser.add_argument("--models-dir", type=Path, default=config.MODELS_DIR)
    parser.add_argument("--metrics-path", type=Path, default=config.METRICS_PATH)
    args = parser.parse_args(argv)

    game_logs = local.read_game_logs(args.data_dir)
    results = run(game_logs, args.models_dir)
    payload = evaluate.write_metrics(
        results, args.metrics_path, dataset_version=local.dataset_fingerprint(args.data_dir)
    )
    print(evaluate.metrics_table(payload))
    print(f"\nwrote {args.metrics_path}")


if __name__ == "__main__":
    main()
