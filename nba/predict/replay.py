"""Replay the daily slate over a whole season using only data available as of each date.

For every date with regular-season games in the season's schedule:
  1. truncate the stored game logs to game_date < date (strict as-of),
  2. build the slate exactly as the nightly job does (roster rule, pending features,
     pinned model),
  3. join the predictions to the actual game logs.

The season MAE is then compared with reports/metrics.json, which scored the same
model on the holdout season restricted to rows with minutes >= MIN_MINUTES and both
baselines defined. The comparison uses that same restricted population; the
unrestricted MAE and the count of actual rows the roster rule never predicted
(e.g. post-trade debuts) are reported alongside.

Products written to replay/<season>/ (and pushed with --push-products):
  replay.json             the full report (also written to reports/replay_<season>.json)
  daily_mae.json          per date: n, model MAE, last-10 baseline MAE per target
  sample_<last-date>.json that date's slate with predictions and actuals side by side
  residuals/<date>.parquet one residual file per replayed date (nightly residual columns)

Usage:
    python -m nba.predict.replay [--season 2025-26] [--schedule-dir data/dump]
                                 [--download-schedule] [--data-dir data/game_logs]
                                 [--models-dir <local>] [--metrics reports/metrics.json]
                                 [--out reports/replay_2025-26.json] [--tolerance 0.05] [--no-pull]
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from nba import config
from nba.ingest import kaggle_daily, kaggle_dump, schedule
from nba.models import evaluate
from nba.predict import model, residuals, slate
from nba.storage import hf, local

TOLERANCE = 0.05


def _mae(frame: pd.DataFrame) -> dict[str, float | None]:
    return residuals.mae(frame)


def replay_season(
    season: str,
    game_logs: pd.DataFrame,
    sched: pd.DataFrame,
    models: model.Models,
    dataset_revision: str,
    dates: list[date] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Per-row residual frame for the season and per-date counts."""
    season_games = sched[sched["season"] == season]
    all_dates = sorted(d.date() for d in season_games["game_date"].unique())
    if dates is not None:
        all_dates = [d for d in all_dates if d in set(dates)]
    frames: list[pd.DataFrame] = []
    per_date: list[dict[str, Any]] = []
    for d in all_dates:
        as_of = pd.Timestamp(d)
        games = schedule.games_on(season_games, d)
        history = game_logs[game_logs["game_date"] < as_of]
        pending = slate.pending_rows(history, games, d)
        if pending.empty:
            per_date.append({"date": d.isoformat(), "n_games": int(len(games)), "n_predicted": 0})
            continue
        # Features depend only on each player's own rows: keep just those players' history.
        players = set(pending["player_id"])
        history = history[history["player_id"].isin(players)]
        preds = slate.score_pending(
            history, pending, models, dataset_revision, generated_at="replay"
        )
        joined = residuals.join_actuals(preds, game_logs)
        frames.append(joined)
        per_date.append(
            {
                "date": d.isoformat(),
                "n_games": int(len(games)),
                "n_predicted": int(len(joined)),
                "n_with_actuals": int(joined["has_actual"].sum()),
            }
        )
    combined = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=list(residuals.RESIDUAL_COLUMNS))
    )
    return combined, {"dates": per_date, "season_game_ids": sorted(season_games["game_id"])}


def unpredicted_actual_rows(
    game_logs: pd.DataFrame, season_game_ids: list[str], predicted: pd.DataFrame
) -> dict[str, int]:
    """Actual rows in the season's games that no replayed prediction covered."""
    played = game_logs[game_logs["game_id"].isin(season_game_ids)]
    keys = ["player_id", "game_id"]
    merged = played.merge(predicted[keys].drop_duplicates(), on=keys, how="left", indicator=True)
    missed = merged[merged["_merge"] == "left_only"]
    return {
        "all": int(len(missed)),
        "minutes_ge_min": int((missed["minutes"] >= config.MIN_MINUTES).sum()),
    }


def daily_mae(combined: pd.DataFrame, season: str, sample_date: str | None) -> dict[str, Any]:
    """Per date: n and MAE per target for the model and the last-10 baseline.

    Both are computed on the same rows: those with actuals and a defined last-10
    baseline for every target.
    """
    base_cols = [f"{t}_mean_last10" for t in config.TARGETS]
    rows = combined[combined["has_actual"] & combined[base_cols].notna().all(axis=1)]
    days = []
    for d, part in rows.groupby("date", sort=True):
        entry: dict[str, Any] = {
            "date": str(d),
            "n": int(len(part)),
            "model": {},
            "baseline_last10": {},
        }
        for t in config.TARGETS:
            entry["model"][t] = float(part[f"resid_{t}"].abs().mean())
            entry["baseline_last10"][t] = float(
                (part[f"actual_{t}"] - part[f"{t}_mean_last10"]).abs().mean()
            )
        days.append(entry)
    return {
        "season": season,
        "population": "rows with actuals and a last-10 baseline for every target",
        "targets": list(config.TARGETS),
        "n_dates": len(days),
        "first_date": days[0]["date"] if days else None,
        "last_date": days[-1]["date"] if days else None,
        "sample_date": sample_date,
        "sample_file": f"sample_{sample_date}.json" if sample_date else None,
        "days": days,
    }


def sample_slate(
    combined: pd.DataFrame, d: str, models: model.Models, dataset_revision: str
) -> dict[str, Any]:
    """One date's replayed slate with predictions and actuals side by side."""
    part = combined[combined["date"] == d].sort_values(["game_id", "team", "player_id"])
    rows = []
    for r in part.itertuples(index=False):
        rows.append(
            {
                "game_id": str(r.game_id),
                "player_id": int(r.player_id),
                "player_name": str(r.player_name),
                "team": str(r.team),
                "opponent": str(r.opponent),
                "home": bool(r.home),
                "pred_pts": round(float(r.pred_pts), 2),
                "pred_reb": round(float(r.pred_reb), 2),
                "pred_ast": round(float(r.pred_ast), 2),
                "actual_pts": None if pd.isna(r.actual_pts) else int(r.actual_pts),
                "actual_reb": None if pd.isna(r.actual_reb) else int(r.actual_reb),
                "actual_ast": None if pd.isna(r.actual_ast) else int(r.actual_ast),
                "minutes": None if pd.isna(r.minutes) else round(float(r.minutes), 1),
                "has_actual": bool(r.has_actual),
            }
        )
    return {
        "date": d,
        "model_revision": models.revision,
        "dataset_revision": dataset_revision,
        "n_games": int(part["game_id"].nunique()),
        "n_players": len(rows),
        "n_with_actuals": int(part["has_actual"].sum()),
        "rows": rows,
    }


def write_products(
    season: str,
    combined: pd.DataFrame,
    report: dict[str, Any],
    models: model.Models,
    dataset_revision: str,
    products_dir: Path,
) -> list[Path]:
    """Write replay/<season>/{replay.json, daily_mae.json, sample_<last-date>.json}."""
    folder = products_dir / season
    folder.mkdir(parents=True, exist_ok=True)
    last_date = str(combined["date"].max()) if len(combined) else None
    written = []
    replay_path = folder / "replay.json"
    replay_path.write_text(json.dumps(report, indent=2) + "\n")
    written.append(replay_path)
    daily = daily_mae(combined, season, last_date)
    daily_path = folder / "daily_mae.json"
    daily_path.write_text(json.dumps(daily, indent=2) + "\n")
    written.append(daily_path)
    if last_date:
        sample_path = folder / f"sample_{last_date}.json"
        sample_path.write_text(
            json.dumps(sample_slate(combined, last_date, models, dataset_revision), indent=2) + "\n"
        )
        written.append(sample_path)
    # One residual file per replayed date, in the nightly residual layout, so the
    # analyst tools and the golden-set evals can read any 2025-26 date.
    res_dir = folder / "residuals"
    res_dir.mkdir(exist_ok=True)
    for d, part in combined.groupby("date", sort=True):
        path = res_dir / f"{d}.parquet"
        part.reset_index(drop=True).to_parquet(path, index=False)
        written.append(path)
    return written


def summarize(
    season: str,
    combined: pd.DataFrame,
    per_date: dict[str, Any],
    game_logs: pd.DataFrame,
    metrics: dict[str, Any],
    models: model.Models,
    dataset_revision: str,
    tolerance: float = TOLERANCE,
) -> dict[str, Any]:
    with_actuals = combined[combined["has_actual"]]
    restricted = with_actuals[with_actuals["in_metrics_population"]]
    reference = {t: metrics["metrics"][t]["model"]["mae"] for t in config.TARGETS}
    mae_restricted = _mae(restricted)
    diffs = {
        t: (None if mae_restricted[t] is None else round(mae_restricted[t] - reference[t], 4))
        for t in config.TARGETS
    }
    passed = all(d is not None and abs(d) <= tolerance for d in diffs.values())
    return {
        "season": season,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_sha": evaluate.git_sha(),
        "model_revision": models.revision,
        "dataset_revision": dataset_revision,
        "reference": {
            "metrics_json_git_sha": metrics.get("git_sha"),
            "metrics_json_dataset": metrics.get("dataset"),
            "model_mae": reference,
            "n": metrics["metrics"][config.TARGETS[0]]["model"]["n"],
        },
        "n_dates": len(per_date["dates"]),
        "n_predicted": int(len(combined)),
        "n_with_actuals": int(len(with_actuals)),
        "n_missing_actuals": int((~combined["has_actual"]).sum()) if len(combined) else 0,
        "n_restricted": int(len(restricted)),
        "mae_restricted": mae_restricted,
        "mae_unrestricted": _mae(with_actuals),
        "unpredicted_actual_rows": unpredicted_actual_rows(
            game_logs, per_date["season_game_ids"], combined
        ),
        "diff_vs_metrics_json": diffs,
        "tolerance": tolerance,
        "passed": passed,
        "per_date": per_date["dates"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--season", default=config.HOLDOUT_SEASON)
    parser.add_argument("--schedule-dir", type=Path, default=config.DUMP_DIR)
    parser.add_argument(
        "--download-schedule", action="store_true", help="fetch schedule + histories from Kaggle"
    )
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    parser.add_argument("--models-dir", type=Path, default=None)
    parser.add_argument("--metrics", type=Path, default=config.METRICS_PATH)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--tolerance", type=float, default=TOLERANCE)
    parser.add_argument("--no-pull", action="store_true")
    parser.add_argument("--products-dir", type=Path, default=config.REPLAY_DIR)
    parser.add_argument(
        "--push-products", action="store_true", help="upload replay/<season>/ to the dataset repo"
    )
    args = parser.parse_args(argv)
    out = args.out or (config.REPORTS_DIR / f"replay_{args.season}.json")

    revision = None if args.no_pull else hf.pull_dataset(args.data_dir)
    game_logs = local.read_game_logs(args.data_dir)
    dataset_revision = revision or local.dataset_fingerprint(args.data_dir)
    if args.download_schedule:
        for name in (kaggle_dump.TEAM_HISTORY_FILE, schedule.schedule_file_name(args.season)):
            if kaggle_daily.download_kaggle_file(name, args.schedule_dir) is None:
                raise FileNotFoundError(f"{name} is not in {config.KAGGLE_DATASET}")
    histories = kaggle_dump.load_team_histories(args.schedule_dir)
    sched = schedule.load_schedule(
        args.schedule_dir / schedule.schedule_file_name(args.season), histories
    )
    models = model.load_from_dir(args.models_dir) if args.models_dir else model.load_from_hub()
    metrics = json.loads(args.metrics.read_text())

    combined, per_date = replay_season(args.season, game_logs, sched, models, dataset_revision)
    report = summarize(
        args.season,
        combined,
        per_date,
        game_logs,
        metrics,
        models,
        dataset_revision,
        args.tolerance,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    products = write_products(
        args.season, combined, report, models, dataset_revision, args.products_dir
    )
    print(f"REPLAY products: {[str(p) for p in products]}")
    if args.push_products:
        sha = hf.push_products(
            args.products_dir.parent, files=products, message=f"Replay {args.season}"
        )
        print(f"REPLAY pushed products at {sha}")
    print(
        f"REPLAY {args.season}: {report['n_dates']} dates, {report['n_predicted']} predictions, "
        f"{report['n_with_actuals']} with actuals, {report['n_restricted']} in metrics population"
    )
    for t in config.TARGETS:
        got, ref = report["mae_restricted"][t], report["reference"]["model_mae"][t]
        diff, unres = report["diff_vs_metrics_json"][t], report["mae_unrestricted"][t]
        print(
            f"REPLAY {t}: restricted MAE {got:.4f} vs metrics.json {ref:.4f} "
            f"(diff {diff:+.4f}); unrestricted {unres:.4f}"
        )
    print(f"REPLAY unpredicted actual rows: {report['unpredicted_actual_rows']}")
    verdict = "PASSED" if report["passed"] else "FAILED"
    print(f"REPLAY {verdict} (tolerance {args.tolerance}); wrote {out}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
