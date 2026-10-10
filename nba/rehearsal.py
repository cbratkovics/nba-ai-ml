"""Rehearse the nightly path for a date with games, without writing anything anywhere public.

The nightly job's warehouse steps (gate, target, incremental build, export, publish) had never
executed before the 2026-27 season because every off-season run ended at the gate
(docs/AE_AUDIT.md §2). This module lets the same workflow run them for real on a synthetic
game night (ADR-0028):

  1. the stored game logs are pulled from Hugging Face (a read);
  2. a dump directory is written with the real TeamHistories.csv, a header-only
     PlayerStatistics.csv (so the daily ingest sees an empty window and changes nothing) and a
     synthetic LeagueSchedule file for the date's season that re-dates the most recent real
     game night to the rehearsal date with 2026-27-style game ids;
  3. `nba.nightly.run` runs with push disabled and no agent key: it writes predictions,
     decisions, the daily report and the drift report under the root, and the summary the
     warehouse gate reads says "slate written", so the workflow's warehouse steps run;
  4. `stage-products` copies those local products into the warehouse root the loader filled
     from Hugging Face, which is what `pull_products` would have done had they been pushed.

Writes to Hugging Face are impossible while NBA_REHEARSAL is set: nba.storage.hf refuses to
build a write client. The publish step runs with --dry-run.

Usage:
    NBA_REHEARSAL=1 python -m nba.rehearsal --date 2026-10-20 [--root .] [--dump-dir data/dump]
                                             [--team-histories <TeamHistories.csv>] [--no-pull]
    python -m nba.rehearsal stage-products [--root .] [--warehouse data/warehouse]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import asdict
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd

from nba import config, nightly
from nba.ingest import kaggle_daily, kaggle_dump, schedule
from nba.predict import model
from nba.storage import hf, local

REHEARSAL_ENV = "NBA_REHEARSAL"
SCHEDULE_COLUMNS = (
    "gameId",
    "gameDateTimeEst",
    "homeTeamId",
    "awayTeamId",
    "gameLabel",
    "gameSubLabel",
)
# Product families the warehouse loader pulls from Hugging Face and the glob per family.
STAGED_FAMILIES: tuple[tuple[str, str], ...] = (
    (config.HF_PREDICTIONS_PREFIX, "*.parquet"),
    (config.HF_RESIDUALS_PREFIX, "*.parquet"),
    (config.HF_DAILY_REPORTS_PREFIX, "*.json"),
    (config.HF_DRIFT_PREFIX, "*.json"),
)


def team_ids(histories: pd.DataFrame, season_year: int) -> dict[str, int]:
    """Abbreviation -> team id for the franchises active in the season (TeamHistories rows)."""
    active = histories[
        (histories["seasonFounded"] <= season_year) & (histories["seasonActiveTill"] >= season_year)
    ]
    return {str(a): int(t) for a, t in zip(active["teamAbbrev"], active["teamId"], strict=True)}


def synthetic_schedule(game_logs: pd.DataFrame, d: date, histories: pd.DataFrame) -> pd.DataFrame:
    """The most recent real game night re-dated to `d`, in the dump's LeagueSchedule layout."""
    season = schedule.season_for_date(d)
    ids = team_ids(histories, schedule.season_start_year(season))
    last_date = game_logs["game_date"].max()
    night = game_logs[game_logs["game_date"] == last_date]
    rows = []
    for i, (_gid, g) in enumerate(sorted(night.groupby("game_id")), start=1):
        home = g.loc[g["home"], "team"]
        away = g.loc[~g["home"], "team"]
        if home.empty or away.empty:
            continue
        rows.append(
            {
                # 22600001 as the dump writes it (no leading zeros); zero-padded to the
                # regular-season id 0022600001 by the schedule loader.
                "gameId": int(f"2{season[2:4]}{i:05d}"),
                "gameDateTimeEst": f"{d.isoformat()} 19:30:00",
                "homeTeamId": ids[str(home.iloc[0])],
                "awayTeamId": ids[str(away.iloc[0])],
                "gameLabel": None,
                "gameSubLabel": None,
            }
        )
    if not rows:
        raise ValueError(f"no games found on {last_date.date()} to re-date")
    return pd.DataFrame(rows, columns=list(SCHEDULE_COLUMNS))


def write_dump(dump_dir: Path, game_logs: pd.DataFrame, d: date, team_histories: Path) -> dict:
    """TeamHistories.csv, a header-only PlayerStatistics.csv and the synthetic schedule."""
    dump_dir.mkdir(parents=True, exist_ok=True)
    target = dump_dir / kaggle_dump.TEAM_HISTORY_FILE
    if team_histories.resolve() != target.resolve():  # the default downloads it into dump_dir
        shutil.copyfile(team_histories, target)
    header = list(kaggle_dump.BOX_SCORE_COLUMNS) + list(kaggle_dump.BOX_SCORE_DERIVED)
    pd.DataFrame(columns=header).to_csv(dump_dir / kaggle_dump.BOX_SCORE_FILE, index=False)
    histories = kaggle_dump.load_team_histories(dump_dir)
    sched = synthetic_schedule(game_logs, d, histories)
    name = schedule.schedule_file_name(schedule.season_for_date(d))
    sched.to_csv(dump_dir / name, index=False)
    return {"schedule_file": name, "n_games": int(len(sched)), "dump_dir": dump_dir.as_posix()}


def stage_products(root: Path, warehouse_root: Path) -> list[Path]:
    """Copy tonight's local product files into the loader's warehouse root."""
    staged = []
    for family, pattern in STAGED_FAMILIES:
        src = root / family
        if not src.is_dir():
            continue
        dest = warehouse_root / family
        dest.mkdir(parents=True, exist_ok=True)
        for p in sorted(src.glob(pattern)):
            shutil.copyfile(p, dest / p.name)
            staged.append(dest / p.name)
    return staged


def run(
    d: date,
    root: Path,
    dump_dir: Path,
    team_histories: Path,
    *,
    pull: bool,
    load_models=model.load_from_hub,
    summary_path: Path | None = None,
) -> nightly.NightlySummary:
    os.environ[REHEARSAL_ENV] = "1"
    config.GROQ_API_KEY = (
        None  # no model calls on a rehearsal night; the brief is agent_unavailable
    )
    data_dir = root / config.DATA_DIR
    if pull:
        rev = hf.pull_dataset(data_dir)
        print(f"REHEARSAL {d}: pulled game logs at {rev} (read only)")
    game_logs = local.read_game_logs(data_dir)
    dump = write_dump(dump_dir, game_logs, d, team_histories)
    print(
        f"REHEARSAL {d}: synthetic {dump['schedule_file']} with {dump['n_games']} games "
        f"in {dump_dir}"
    )
    summary = nightly.run(d, push=False, root=root, local_dump=dump_dir, load_models=load_models)
    summary.lines.insert(
        0, f"REHEARSAL {d}: no pushes; {REHEARSAL_ENV}=1 blocks Hugging Face writes"
    )
    path = summary_path or nightly.SUMMARY_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(summary), indent=2) + "\n")
    print(f"REHEARSAL {d}: summary written to {path}")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command")
    p_stage = sub.add_parser("stage-products", help="copy local products into the warehouse root")
    p_stage.add_argument("--root", type=Path, default=Path("."))
    p_stage.add_argument("--warehouse", type=Path, default=Path("data") / "warehouse")
    parser.add_argument("--date", type=date.fromisoformat, default=None)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--dump-dir", type=Path, default=config.DUMP_DIR)
    parser.add_argument("--team-histories", type=Path, default=None, help="default: download it")
    parser.add_argument("--no-pull", action="store_true", help="use data/game_logs as-is")
    parser.add_argument("--summary", type=Path, default=nightly.SUMMARY_PATH)
    args = parser.parse_args(argv)
    if args.command == "stage-products":
        staged = stage_products(args.root, args.warehouse)
        for p in staged:
            print(f"STAGED {p}")
        print(f"REHEARSAL staged {len(staged)} product files into {args.warehouse}")
        return 0
    d = args.date or datetime.now(UTC).date()
    histories = args.team_histories
    if histories is None:
        got = kaggle_daily.download_kaggle_file(kaggle_dump.TEAM_HISTORY_FILE, args.dump_dir)
        if got is None:
            raise FileNotFoundError(
                f"{kaggle_dump.TEAM_HISTORY_FILE} is not in {config.KAGGLE_DATASET}"
            )
        histories = got
    run(d, args.root, args.dump_dir, histories, pull=not args.no_pull, summary_path=args.summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
