"""Residuals of stored predictions against actual game logs.

For a date D (normally yesterday):
  1. read predictions/D.parquet (written by the slate); if absent, report and exit 0;
  2. left-join actual game logs on (player_id, game_id); predicted rows with no game
     log are counted as missing actuals, split into "game not ingested yet" and
     "player did not play", never dropped silently;
  3. write residuals/D.parquet with predictions, actuals, and residuals;
  4. append (or replace) the line for D in predictions/rolling_metrics.json, a JSON
     Lines file with per-day counts, MAE per target, and the rolling 30-day MAE
     computed from every residual file within the window.

MAE is reported unrestricted (every row with actuals) and restricted to the
population reports/metrics.json uses (minutes >= MIN_MINUTES and both baselines
present), so daily numbers are comparable with the published holdout figures.

Usage:
    python -m nba.predict.residuals [--date YYYY-MM-DD] [--data-dir data/game_logs]
                                    [--predictions-dir predictions] [--residuals-dir residuals]
                                    [--no-pull]
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from nba import config
from nba.storage import hf, local

ROLLING_FILE = "rolling_metrics.json"
ROLLING_WINDOW_DAYS = 30
RESIDUAL_COLUMNS: tuple[str, ...] = (
    "date",
    "game_id",
    "player_id",
    "player_name",
    "team",
    "opponent",
    "home",
    "pred_pts",
    "pred_reb",
    "pred_ast",
    "actual_pts",
    "actual_reb",
    "actual_ast",
    "resid_pts",
    "resid_reb",
    "resid_ast",
    "minutes",
    "pts_mean_last10",
    "reb_mean_last10",
    "ast_mean_last10",
    "has_actual",
    "game_ingested",
    "in_metrics_population",
    "model_revision",
    "dataset_revision",
)


@dataclass
class ResidualResult:
    date: date
    residuals: pd.DataFrame
    line: dict[str, Any]
    message: str


def mae(frame: pd.DataFrame) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for t in config.TARGETS:
        out[t] = float(frame[f"resid_{t}"].abs().mean()) if len(frame) else None
    return out


def join_actuals(predictions: pd.DataFrame, game_logs: pd.DataFrame) -> pd.DataFrame:
    """Predictions with actual stats, residuals, and the missing-actual flags."""
    keys = ["player_id", "game_id"]
    actual_cols = {"pts": "actual_pts", "reb": "actual_reb", "ast": "actual_ast"}
    actuals = game_logs[keys + ["minutes", *actual_cols]].rename(columns=actual_cols)
    ingested_games = set(game_logs["game_id"])
    df = predictions.merge(actuals, on=keys, how="left", indicator=True)
    df["has_actual"] = df["_merge"] == "both"
    df["game_ingested"] = df["game_id"].isin(ingested_games)
    for t in config.TARGETS:
        df[f"resid_{t}"] = df[f"actual_{t}"] - df[f"pred_{t}"]
    both_baselines = df[[f"{t}_mean_last10" for t in config.TARGETS]].notna().all(axis=1) & df[
        [f"{t}_mean_season" for t in config.TARGETS]
    ].notna().all(axis=1)
    df["in_metrics_population"] = (
        df["has_actual"] & (df["minutes"] >= config.MIN_MINUTES) & both_baselines
    )
    return df[list(RESIDUAL_COLUMNS)].reset_index(drop=True)


def unpredicted_actuals(predictions: pd.DataFrame, game_logs: pd.DataFrame) -> int:
    """Actual rows in the predicted games that no prediction covered (e.g. post-trade debuts)."""
    games = set(predictions["game_id"])
    played = game_logs[game_logs["game_id"].isin(games)]
    keys = ["player_id", "game_id"]
    covered = played.merge(predictions[keys], on=keys, how="left", indicator=True)
    return int((covered["_merge"] == "left_only").sum())


def rolling_mae(residuals_dir: Path, end: date, window_days: int = ROLLING_WINDOW_DAYS) -> dict:
    """MAE per target over every residual file dated in (end - window, end]."""
    start = end - timedelta(days=window_days)
    frames = []
    for path in sorted(residuals_dir.glob("*.parquet")):
        try:
            d = date.fromisoformat(path.stem)
        except ValueError:
            continue
        if start < d <= end:
            frames.append(pd.read_parquet(path))
    if not frames:
        return {"days": 0, "n": 0, "mae": dict.fromkeys(config.TARGETS)}
    allres = pd.concat(frames, ignore_index=True)
    with_actuals = allres[allres["has_actual"]]
    return {
        "days": len(frames),
        "n": int(len(with_actuals)),
        "mae": mae(with_actuals),
        "n_restricted": int(with_actuals["in_metrics_population"].sum()),
        "mae_restricted": mae(with_actuals[with_actuals["in_metrics_population"]]),
    }


def compute(
    d: date,
    predictions: pd.DataFrame,
    game_logs: pd.DataFrame,
    residuals_dir: Path,
) -> ResidualResult:
    df = join_actuals(predictions, game_logs)
    with_actuals = df[df["has_actual"]]
    restricted = with_actuals[with_actuals["in_metrics_population"]]
    missing = df[~df["has_actual"]]
    residuals_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(residuals_dir / f"{d.isoformat()}.parquet", index=False)
    line = {
        "date": d.isoformat(),
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "n_predicted": int(len(df)),
        "n_games_predicted": int(df["game_id"].nunique()),
        "n_games_ingested": int(df.loc[df["game_ingested"], "game_id"].nunique()),
        "n_with_actuals": int(len(with_actuals)),
        "n_missing_actuals": int(len(missing)),
        "n_missing_game_not_ingested": int((~missing["game_ingested"]).sum()),
        "n_missing_player_did_not_play": int(missing["game_ingested"].sum()),
        "n_unpredicted_actuals": unpredicted_actuals(predictions, game_logs),
        "mae": mae(with_actuals),
        "n_restricted": int(len(restricted)),
        "mae_restricted": mae(restricted),
        "rolling_30d": rolling_mae(residuals_dir, d),
        "model_revision": str(predictions["model_revision"].iloc[0]) if len(predictions) else None,
    }
    message = (
        f"RESIDUALS {d}: {line['n_predicted']} predicted, {line['n_with_actuals']} with actuals, "
        f"{line['n_missing_actuals']} missing ({line['n_missing_game_not_ingested']} game not "
        f"ingested, {line['n_missing_player_did_not_play']} did not play); "
        f"MAE pts={_fmt(line['mae']['pts'])} reb={_fmt(line['mae']['reb'])} "
        f"ast={_fmt(line['mae']['ast'])}; rolling 30d n={line['rolling_30d']['n']}"
    )
    return ResidualResult(d, df, line, message)


def _fmt(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.3f}"


def append_rolling_line(predictions_dir: Path, line: dict[str, Any]) -> Path:
    """Append the day's line to rolling_metrics.json (JSON Lines); replace an existing date."""
    predictions_dir.mkdir(parents=True, exist_ok=True)
    path = predictions_dir / ROLLING_FILE
    existing = []
    if path.exists():
        existing = [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
    kept = [ln for ln in existing if ln.get("date") != line["date"]]
    kept.append(line)
    kept.sort(key=lambda ln: ln["date"])
    path.write_text("".join(json.dumps(ln) + "\n" for ln in kept))
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--date", type=date.fromisoformat, default=None, help="default: yesterday")
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    parser.add_argument("--predictions-dir", type=Path, default=config.PREDICTIONS_DIR)
    parser.add_argument("--residuals-dir", type=Path, default=config.RESIDUALS_DIR)
    parser.add_argument("--no-pull", action="store_true", help="use local dirs as-is")
    args = parser.parse_args(argv)
    d = args.date or (datetime.now(UTC).date() - timedelta(days=1))

    if not args.no_pull:
        hf.pull_dataset(args.data_dir)
        hf.pull_products(args.predictions_dir.parent)
    pred_path = args.predictions_dir / f"{d.isoformat()}.parquet"
    if not pred_path.exists():
        print(f"RESIDUALS {d}: no predictions file {pred_path}; nothing to score")
        return 0
    predictions = pd.read_parquet(pred_path)
    game_logs = local.read_game_logs(args.data_dir)
    result = compute(d, predictions, game_logs, args.residuals_dir)
    path = append_rolling_line(args.predictions_dir, result.line)
    print(result.message)
    print(f"RESIDUALS wrote {args.residuals_dir / (d.isoformat() + '.parquet')} and updated {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
