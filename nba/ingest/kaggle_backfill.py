"""Backfill game logs from the Eoin Moore Kaggle dump.

Dataset: https://www.kaggle.com/datasets/eoinamoore/historical-nba-data-and-player-box-scores
(CC0). Download it manually and pass the extracted directory with --kaggle-dir.
No Kaggle credentials are used or stored here.

Two files are read:
  * PlayerStatistics.csv  (one row per player per game: the box score)
  * Games.csv             (one row per game: the schedule, with date and game type)

Rows are mapped into the canonical schema in `nba.schema` and written as one
Parquet file per season. This module and `nba.storage` do not import
`nba.models`, so the backfill and push run without LightGBM installed.

Usage:
    python -m nba.ingest.kaggle_backfill --kaggle-dir <path> [--out-dir data/game_logs] [--push]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from nba_api.stats.static import teams as nba_teams

from nba import config, schema
from nba.storage import hf, local

BOX_SCORE_FILE = "PlayerStatistics.csv"
SCHEDULE_FILE = "Games.csv"

# Kaggle column -> canonical column. Names not listed here are derived below.
BOX_SCORE_COLUMNS: dict[str, str] = {
    "personId": "player_id",
    "gameId": "game_id",
    "numMinutes": "minutes",
    "points": "pts",
    "reboundsTotal": "reb",
    "assists": "ast",
    "fieldGoalsMade": "fgm",
    "fieldGoalsAttempted": "fga",
    "threePointersMade": "fg3m",
    "threePointersAttempted": "fg3a",
    "freeThrowsMade": "ftm",
    "freeThrowsAttempted": "fta",
    "reboundsOffensive": "oreb",
    "reboundsDefensive": "dreb",
    "steals": "stl",
    "blocks": "blk",
    "turnovers": "tov",
    "foulsPersonal": "pf",
    "plusMinusPoints": "plus_minus",
}
# Kaggle columns used to derive canonical columns.
BOX_SCORE_DERIVED: tuple[str, ...] = (
    "firstName",
    "lastName",
    "playerteamName",
    "opponentteamName",
    "home",
)
SCHEDULE_COLUMNS: dict[str, str] = {
    "gameId": "game_id",
    "gameDate": "game_date",
    "gameType": "game_type",
}

COUNTING_STATS: tuple[str, ...] = (
    "pts",
    "reb",
    "ast",
    "fgm",
    "fga",
    "fg3m",
    "fg3a",
    "ftm",
    "fta",
    "oreb",
    "dreb",
    "stl",
    "blk",
    "tov",
    "pf",
    "plus_minus",
)

GAME_ID_WIDTH = 10  # nba.com game ids are zero-padded to 10 characters


def season_from_date(ts: pd.Timestamp) -> str:
    """Season label for a game date, e.g. 2024-11-01 -> '2024-25'."""
    start_year = ts.year if ts.month >= 8 else ts.year - 1
    return f"{start_year}-{(start_year + 1) % 100:02d}"


def team_abbreviations() -> dict[str, str]:
    """Team nickname -> abbreviation from nba_api's bundled static table (no HTTP)."""
    return {t["nickname"]: t["abbreviation"] for t in nba_teams.get_teams()}


def normalize_game_id(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip().str.zfill(GAME_ID_WIDTH)


def load_box_scores(kaggle_dir: Path) -> pd.DataFrame:
    return pd.read_csv(kaggle_dir / BOX_SCORE_FILE, low_memory=False)


def load_schedule(kaggle_dir: Path) -> pd.DataFrame:
    return pd.read_csv(kaggle_dir / SCHEDULE_FILE, low_memory=False)


def _require(df: pd.DataFrame, columns: list[str], what: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"{what} is missing expected columns: {missing}")


def map_to_schema(
    box: pd.DataFrame,
    sched: pd.DataFrame,
    seasons: tuple[str, ...] = config.SEASONS,
    game_types: tuple[str, ...] = config.GAME_TYPES,
) -> pd.DataFrame:
    """Map raw Kaggle frames to the canonical schema and validate."""
    _require(box, list(BOX_SCORE_COLUMNS) + list(BOX_SCORE_DERIVED), BOX_SCORE_FILE)
    _require(sched, list(SCHEDULE_COLUMNS), SCHEDULE_FILE)

    games = sched.rename(columns=SCHEDULE_COLUMNS)[list(SCHEDULE_COLUMNS.values())].copy()
    games["game_id"] = normalize_game_id(games["game_id"])
    games["game_date"] = pd.to_datetime(games["game_date"]).dt.tz_localize(None).dt.normalize()
    games = games.drop_duplicates(subset="game_id")

    wanted_types = {t.strip().lower() for t in game_types}
    games = games[games["game_type"].astype("string").str.strip().str.lower().isin(wanted_types)]

    df = box.rename(columns=BOX_SCORE_COLUMNS).copy()
    df["game_id"] = normalize_game_id(df["game_id"])
    df = df.merge(games[["game_id", "game_date"]], on="game_id", how="inner")

    df["season"] = df["game_date"].map(season_from_date)
    df = df[df["season"].isin(seasons)]

    # Players who did not play have no minutes; they are not game logs.
    df = df[df["minutes"].notna() & (df["minutes"] > 0)]

    df["player_name"] = (
        df["firstName"].astype("string").str.strip()
        + " "
        + df["lastName"].astype("string").str.strip()
    )

    abbrev = team_abbreviations()
    for src, dst in (("playerteamName", "team"), ("opponentteamName", "opponent")):
        names = df[src].astype("string").str.strip()
        unknown = sorted(set(names.dropna().unique()) - set(abbrev))
        if unknown:
            raise KeyError(f"unknown team nicknames in {src}: {unknown}")
        df[dst] = names.map(abbrev)

    df["home"] = pd.to_numeric(df["home"]).astype("int64").astype("bool")
    for col in COUNTING_STATS:
        df[col] = pd.to_numeric(df[col]).fillna(0).round().astype("int64")
    df["source"] = config.KAGGLE_SOURCE

    out = schema.coerce(df)
    out = out.sort_values(["game_date", "game_id", "player_id"]).reset_index(drop=True)
    return schema.validate(out)


def backfill(kaggle_dir: Path, out_dir: Path) -> dict[str, Path]:
    df = map_to_schema(load_box_scores(kaggle_dir), load_schedule(kaggle_dir))
    return local.write_per_season(df, out_dir)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--kaggle-dir", type=Path, required=True, help="extracted Kaggle dataset directory"
    )
    parser.add_argument(
        "--out-dir", type=Path, default=config.DATA_DIR, help="where to write Parquet"
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help=(
            f"after writing, upload the Parquet files to {config.HF_DATASET_REPO} (needs HF_TOKEN)"
        ),
    )
    args = parser.parse_args(argv)
    written = backfill(args.kaggle_dir, args.out_dir)
    for season, path in written.items():
        rows = pd.read_parquet(path).shape[0]
        print(f"{season}: {rows} rows -> {path}")
    if args.push:
        sha = hf.push_dataset(args.out_dir)
        print(f"pushed to https://huggingface.co/datasets/{config.HF_DATASET_REPO} at {sha}")


if __name__ == "__main__":
    main()
