"""Score every rostered player for the games on a date.

For a date D:
  1. read the schedule file for D's season from the dump directory; if it does not
     exist the outcome is NO_SCHEDULE, if it has no regular-season games on D the
     outcome is NO_GAMES (both exit 0 with one clear log line);
  2. for each team playing, take every player who appeared in any of the team's last
     ROSTER_LOOKBACK_GAMES games before D (from the stored game logs);
  3. build as-of features for those (player, game) pairs using only games before D;
  4. score with the published models pinned to MODEL_REVISION;
  5. write predictions/D.parquet and predictions/latest.json.

Usage:
    python -m nba.predict.slate [--date YYYY-MM-DD] [--schedule-dir data/dump]
                                [--data-dir data/game_logs] [--predictions-dir predictions]
                                [--models-dir <local dir>] [--no-pull]
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from enum import StrEnum
from pathlib import Path

import pandas as pd

from nba import config
from nba.features import asof
from nba.ingest import kaggle_dump, schedule
from nba.predict import model
from nba.storage import hf, local

LATEST_FILE = "latest.json"
OUTPUT_COLUMNS: tuple[str, ...] = (
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
    "pts_mean_last10",
    "reb_mean_last10",
    "ast_mean_last10",
    "pts_mean_season",
    "reb_mean_season",
    "ast_mean_season",
    "games_played_season",
    "model_revision",
    "dataset_revision",
    "generated_at",
)


class SlateStatus(StrEnum):
    OK = "ok"
    NO_SCHEDULE = "no_schedule"
    NO_GAMES = "no_games"


@dataclass
class SlateResult:
    status: SlateStatus
    date: date
    message: str
    predictions: pd.DataFrame | None = None
    n_games: int = 0
    n_players: int = 0


def roster(
    game_logs: pd.DataFrame,
    team: str,
    as_of: pd.Timestamp,
    lookback: int = config.ROSTER_LOOKBACK_GAMES,
) -> pd.DataFrame:
    """Players who appeared in any of the team's last `lookback` games before `as_of`."""
    hist = game_logs[(game_logs["team"] == team) & (game_logs["game_date"] < as_of)]
    last_games = (
        hist[["game_id", "game_date"]]
        .drop_duplicates("game_id")
        .sort_values(["game_date", "game_id"], ascending=False)
        .head(lookback)["game_id"]
    )
    recent = hist[hist["game_id"].isin(last_games)].sort_values(["game_date", "game_id"])
    names = recent.groupby("player_id")["player_name"].last()
    return names.rename("player_name").reset_index()


def pending_rows(game_logs: pd.DataFrame, games: pd.DataFrame, d: date) -> pd.DataFrame:
    """One pending row per rostered player per game on the date."""
    as_of = pd.Timestamp(d)
    season = kaggle_dump.season_from_date(as_of)
    parts = []
    for g in games.itertuples(index=False):
        for team, opponent, home in ((g.home, g.away, True), (g.away, g.home, False)):
            players = roster(game_logs, team, as_of)
            if players.empty:
                continue
            parts.append(
                players.assign(
                    game_id=g.game_id,
                    game_date=as_of,
                    season=season,
                    team=team,
                    opponent=opponent,
                    home=home,
                )
            )
    if not parts:
        return pd.DataFrame(columns=list(asof.PENDING_COLUMNS))
    pending = pd.concat(parts, ignore_index=True)[list(asof.PENDING_COLUMNS)]
    # A traded player can sit in the last-10 window of two teams. Keep them only on the
    # team they most recently played for, so each (player, game) appears once.
    latest_team = (
        game_logs[game_logs["game_date"] < as_of]
        .sort_values(["game_date", "game_id"])
        .groupby("player_id")["team"]
        .last()
    )
    current = pending["player_id"].map(latest_team)
    return pending[pending["team"] == current].reset_index(drop=True)


def score_pending(
    game_logs: pd.DataFrame,
    pending: pd.DataFrame,
    models: model.Models,
    dataset_revision: str,
    generated_at: str | None = None,
) -> pd.DataFrame:
    """Features + predictions for pending rows, in OUTPUT_COLUMNS order."""
    generated_at = generated_at or datetime.now(UTC).isoformat(timespec="seconds")
    features = asof.features_for_pending(game_logs, pending)
    preds = model.predict(models, features)
    out = pd.concat([features, preds], axis=1)
    out = out.merge(pending[["player_id", "game_id", "player_name"]], on=["player_id", "game_id"])
    out["date"] = out["game_date"].dt.date.astype("string")
    out["model_revision"] = models.revision
    out["dataset_revision"] = dataset_revision
    out["generated_at"] = generated_at
    out["home"] = out["home"].astype("bool")
    return (
        out[list(OUTPUT_COLUMNS)]
        .sort_values(["game_id", "team", "player_id"])
        .reset_index(drop=True)
    )


def run_slate(
    d: date,
    game_logs: pd.DataFrame,
    schedule_dir: Path,
    load_models: Callable[[], model.Models],
    dataset_revision: str,
    histories: pd.DataFrame | None = None,
) -> SlateResult:
    """Build and score the slate for one date.

    `load_models` is only called when the date has games, so NO_SCHEDULE and NO_GAMES
    outcomes never touch the model repo.
    """
    season = schedule.season_for_date(d)
    path = schedule.schedule_path(schedule_dir, d)
    if path is None:
        name = schedule.schedule_file_name(season)
        return SlateResult(
            SlateStatus.NO_SCHEDULE,
            d,
            f"SLATE {d}: no schedule file for season {season} ({name} not found in {schedule_dir})",
        )
    if histories is None:
        histories = kaggle_dump.load_team_histories(schedule_dir)
    sched = schedule.load_schedule(path, histories)
    games = schedule.games_on(sched, d)
    if games.empty:
        return SlateResult(
            SlateStatus.NO_GAMES,
            d,
            f"SLATE {d}: no games on this date ({path.name} lists {len(sched)} "
            f"regular-season games, none on {d})",
        )
    models = load_models()
    pending = pending_rows(game_logs, games, d)
    predictions = score_pending(game_logs, pending, models, dataset_revision)
    return SlateResult(
        SlateStatus.OK,
        d,
        f"SLATE {d}: {len(games)} games, {len(predictions)} player predictions",
        predictions=predictions,
        n_games=int(len(games)),
        n_players=int(len(predictions)),
    )


def write_outputs(result: SlateResult, predictions_dir: Path) -> tuple[Path, Path]:
    """Write predictions/<date>.parquet and predictions/latest.json."""
    assert result.status is SlateStatus.OK and result.predictions is not None
    predictions_dir.mkdir(parents=True, exist_ok=True)
    parquet = predictions_dir / f"{result.date.isoformat()}.parquet"
    result.predictions.to_parquet(parquet, index=False)
    p = result.predictions
    latest = {
        "date": result.date.isoformat(),
        "model_revision": str(p["model_revision"].iloc[0]),
        "dataset_revision": str(p["dataset_revision"].iloc[0]),
        "generated_at": str(p["generated_at"].iloc[0]),
        "n_games": result.n_games,
        "n_players": result.n_players,
        "predictions": [
            {
                "player_id": int(r.player_id),
                "player_name": str(r.player_name),
                "team": str(r.team),
                "opponent": str(r.opponent),
                "home": bool(r.home),
                "pred_pts": round(float(r.pred_pts), 2),
                "pred_reb": round(float(r.pred_reb), 2),
                "pred_ast": round(float(r.pred_ast), 2),
            }
            for r in p.itertuples(index=False)
        ],
    }
    latest_path = predictions_dir / LATEST_FILE
    latest_path.write_text(json.dumps(latest, indent=2) + "\n")
    return parquet, latest_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--date", type=date.fromisoformat, default=None, help="slate date")
    parser.add_argument("--schedule-dir", type=Path, default=config.DUMP_DIR)
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    parser.add_argument("--predictions-dir", type=Path, default=config.PREDICTIONS_DIR)
    parser.add_argument("--models-dir", type=Path, default=None, help="local models, not HF")
    parser.add_argument("--no-pull", action="store_true", help="use data-dir as-is")
    args = parser.parse_args(argv)
    d = args.date or datetime.now(UTC).date()

    revision = None if args.no_pull else hf.pull_dataset(args.data_dir)
    game_logs = local.read_game_logs(args.data_dir)
    dataset_revision = revision or local.dataset_fingerprint(args.data_dir)

    def load_models() -> model.Models:
        if args.models_dir:
            return model.load_from_dir(args.models_dir)
        return model.load_from_hub()

    result = run_slate(d, game_logs, args.schedule_dir, load_models, dataset_revision)
    print(result.message)
    if result.status is not SlateStatus.OK:
        return 0
    parquet, latest = write_outputs(result, args.predictions_dir)
    print(f"SLATE wrote {parquet} and {latest} (dataset {dataset_revision})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
