"""Backfill game logs from the Eoin Moore Kaggle dump (version 515).

Dataset: https://www.kaggle.com/datasets/eoinamoore/historical-nba-data-and-player-box-scores
(CC0). Download it manually and pass the extracted directory with --kaggle-dir.
No Kaggle credentials are used or stored here.

Two files are read:
  * PlayerStatistics.csv  one row per player per game (box score), 1947 onward.
                          Streamed with pyarrow and filtered to gameDate >= BACKFILL_START
                          before any pandas frame is built.
  * TeamHistories.csv     teamId, teamAbbrev, seasonFounded, seasonActiveTill. The row
                          active for the game's season supplies the abbreviation.

Games.csv and the LeagueSchedule files are not used.

Rows are mapped into the canonical schema in `nba.schema` and written as one
Parquet file per season. This module and `nba.storage` do not import
`nba.models`, so the backfill and push run without LightGBM installed.

Usage:
    python -m nba.ingest.kaggle_backfill --kaggle-dir <path> [--out-dir data/game_logs]
                                         [--push] [--show-player 203999]
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv

from nba import config, schema
from nba.storage import hf, local

BOX_SCORE_FILE = "PlayerStatistics.csv"
TEAM_HISTORY_FILE = "TeamHistories.csv"

# PlayerStatistics.csv column -> canonical column (or intermediate name).
BOX_SCORE_COLUMNS: dict[str, str] = {
    "gameId": "game_id",
    "gameDate": "game_date",
    "personId": "player_id",
    "playerteamId": "team_id",
    "opponentteamId": "opponent_id",
    "home": "home",
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
# PlayerStatistics.csv columns used to derive canonical columns.
BOX_SCORE_DERIVED: tuple[str, ...] = ("firstName", "lastName", "gameType")
TEAM_HISTORY_COLUMNS: tuple[str, ...] = (
    "teamId",
    "teamAbbrev",
    "seasonFounded",
    "seasonActiveTill",
)

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

# First game date kept when streaming the box-score file: October of the first season.
BACKFILL_START = pd.Timestamp(f"{config.SEASONS[0][:4]}-10-01")

# pyarrow types for the streamed read. Stats are float64 because DNP rows are empty.
_ID_COLUMNS: tuple[str, ...] = ("gameId", "gameDate", "personId", "playerteamId", "opponentteamId")
_BOX_SCORE_TYPES: dict[str, pa.DataType] = {
    "gameId": pa.string(),
    "gameDate": pa.string(),
    "personId": pa.int64(),
    "playerteamId": pa.int64(),
    "opponentteamId": pa.int64(),
    "firstName": pa.string(),
    "lastName": pa.string(),
    "gameType": pa.string(),
    **{c: pa.float64() for c in BOX_SCORE_COLUMNS if c not in _ID_COLUMNS},
}


def season_from_date(ts: pd.Timestamp) -> str:
    """Season label for a game date: start year is the year if month >= 10, else year - 1."""
    start_year = ts.year if ts.month >= 10 else ts.year - 1
    return f"{start_year}-{(start_year + 1) % 100:02d}"


def season_start_year(season: str) -> int:
    return int(season[:4])


def normalize_game_id(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip().str.zfill(GAME_ID_WIDTH)


def _require(columns: Iterable[str], needed: Iterable[str], what: str) -> None:
    have = set(columns)
    missing = [c for c in needed if c not in have]
    if missing:
        raise KeyError(f"{what} is missing expected columns: {missing}")


def load_box_scores(kaggle_dir: Path, start: pd.Timestamp = BACKFILL_START) -> pd.DataFrame:
    """Stream PlayerStatistics.csv and keep only rows with gameDate >= start."""
    path = kaggle_dir / BOX_SCORE_FILE
    needed = list(BOX_SCORE_COLUMNS) + list(BOX_SCORE_DERIVED)
    # Open a streaming reader on a small first block just to inspect the header.
    probe = pacsv.open_csv(path, read_options=pacsv.ReadOptions(block_size=1 << 20))
    _require(probe.schema.names, needed, BOX_SCORE_FILE)
    del probe
    reader = pacsv.open_csv(
        path,
        read_options=pacsv.ReadOptions(block_size=64 << 20),
        convert_options=pacsv.ConvertOptions(include_columns=needed, column_types=_BOX_SCORE_TYPES),
    )
    parts: list[pd.DataFrame] = []
    for batch in reader:
        df = batch.to_pandas()
        dates = pd.to_datetime(df["gameDate"]).dt.tz_localize(None)
        keep = dates >= start
        if keep.any():
            df = df[keep].copy()
            df["gameDate"] = dates[keep]
            parts.append(df)
    if not parts:
        raise ValueError(f"no rows in {path} with gameDate >= {start.date()}")
    return pd.concat(parts, ignore_index=True)


def load_team_histories(kaggle_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(kaggle_dir / TEAM_HISTORY_FILE)
    _require(df.columns, TEAM_HISTORY_COLUMNS, TEAM_HISTORY_FILE)
    out = df[list(TEAM_HISTORY_COLUMNS)].copy()
    out["teamId"] = pd.to_numeric(out["teamId"]).astype("int64")
    out["teamAbbrev"] = out["teamAbbrev"].astype("string").str.strip()
    out["seasonFounded"] = pd.to_numeric(out["seasonFounded"]).astype("int64")
    # Open-ended histories have no end season.
    out["seasonActiveTill"] = pd.to_numeric(out["seasonActiveTill"]).fillna(9999).astype("int64")
    return out


def team_abbreviations(
    team_ids: pd.Series, season_years: pd.Series, histories: pd.DataFrame
) -> pd.Series:
    """Abbreviation for each (team id, season start year), from the history row active then."""
    pairs = pd.DataFrame({"teamId": team_ids.to_numpy(), "year": season_years.to_numpy()})
    unique = pairs.drop_duplicates().reset_index(drop=True)
    merged = unique.merge(histories, on="teamId", how="left")
    is_active = (merged["seasonFounded"] <= merged["year"]) & (
        merged["year"] <= merged["seasonActiveTill"]
    )
    active = merged[is_active]
    counts = active.groupby(["teamId", "year"]).size().rename("n").reset_index()
    matched = unique.merge(counts, on=["teamId", "year"], how="left")
    unmatched = matched[matched["n"].isna()][["teamId", "year"]].to_dict("records")
    if unmatched:
        raise KeyError(f"no active {TEAM_HISTORY_FILE} row for: {unmatched}")
    ambiguous = matched[matched["n"] > 1][["teamId", "year"]].to_dict("records")
    if ambiguous:
        raise KeyError(f"multiple active {TEAM_HISTORY_FILE} rows for: {ambiguous}")
    lookup = active[["teamId", "year", "teamAbbrev"]]
    abbrev = pairs.merge(lookup, on=["teamId", "year"], how="left")["teamAbbrev"]
    return abbrev.astype("string").set_axis(team_ids.index)


def game_type_counts(box: pd.DataFrame) -> pd.Series:
    """Distinct gameType values with row counts, most common first."""
    return box["gameType"].astype("string").str.strip().value_counts(dropna=False)


def map_to_schema(
    box: pd.DataFrame,
    histories: pd.DataFrame,
    seasons: tuple[str, ...] = config.SEASONS,
    game_types: tuple[str, ...] = config.GAME_TYPES,
) -> pd.DataFrame:
    """Map the raw box-score frame to the canonical schema and validate."""
    _require(box.columns, list(BOX_SCORE_COLUMNS) + list(BOX_SCORE_DERIVED), BOX_SCORE_FILE)

    present = set(game_type_counts(box).index.dropna())
    absent = [t for t in game_types if t not in present]
    if absent:
        raise ValueError(f"gameType values {absent} not found; present values: {sorted(present)}")

    kept = box["gameType"].astype("string").str.strip().isin(game_types)
    df = box[kept].rename(columns=BOX_SCORE_COLUMNS).copy()
    df["game_id"] = normalize_game_id(df["game_id"])
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.tz_localize(None).dt.normalize()
    df["season"] = df["game_date"].map(season_from_date)
    df = df[df["season"].isin(seasons)]

    # Players who did not play have no minutes; they are not game logs.
    df = df[df["minutes"].notna() & (df["minutes"] > 0)].copy()

    first = df["firstName"].astype("string").str.strip()
    last = df["lastName"].astype("string").str.strip()
    df["player_name"] = first + " " + last
    years = df["season"].map(season_start_year)
    df["team"] = team_abbreviations(df["team_id"], years, histories)
    df["opponent"] = team_abbreviations(df["opponent_id"], years, histories)

    df["home"] = pd.to_numeric(df["home"]).astype("int64").astype("bool")
    for col in COUNTING_STATS:
        df[col] = pd.to_numeric(df[col]).fillna(0).round().astype("int64")
    df["source"] = config.KAGGLE_SOURCE

    out = schema.coerce(df)
    out = out.sort_values(["game_date", "game_id", "player_id"]).reset_index(drop=True)
    return schema.validate(out)


def backfill(kaggle_dir: Path, out_dir: Path) -> dict[str, Path]:
    df = map_to_schema(load_box_scores(kaggle_dir), load_team_histories(kaggle_dir))
    return local.write_per_season(df, out_dir)


def season_summary(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("season").agg(
        rows=("game_id", "size"), first_game=("game_date", "min"), last_game=("game_date", "max")
    )


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
    parser.add_argument(
        "--show-player", type=int, default=None, help="print this player's last five rows"
    )
    args = parser.parse_args(argv)

    box = load_box_scores(args.kaggle_dir)
    print(f"{BOX_SCORE_FILE}: {len(box)} rows with gameDate >= {BACKFILL_START.date()}")
    print("distinct gameType values:")
    print(game_type_counts(box).to_string())
    print(f"keeping gameType in {list(config.GAME_TYPES)}")

    df = map_to_schema(box, load_team_histories(args.kaggle_dir))
    written = local.write_per_season(df, args.out_dir)
    print(season_summary(df).to_string())
    for season, path in written.items():
        print(f"{season} -> {path}")

    if args.show_player is not None:
        rows = df[df["player_id"] == args.show_player].sort_values("game_date").tail(5)
        print(f"\nlast five rows for player_id {args.show_player}:")
        print(rows.to_string(index=False) if not rows.empty else "(no rows)")

    if args.push:
        sha = hf.push_dataset(args.out_dir)
        print(f"pushed to https://huggingface.co/datasets/{config.HF_DATASET_REPO} at {sha}")


if __name__ == "__main__":
    main()
