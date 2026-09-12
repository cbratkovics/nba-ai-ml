"""Parsing and mapping rules for the Eoin Moore Kaggle dump (version 515).

Shared by the one-off backfill (`nba.ingest.kaggle_backfill`) and the daily
incremental ingest (`nba.ingest.kaggle_daily`). Neither duplicates these rules.

Files understood:
  * PlayerStatistics.csv  one row per player per game (box score), 1947 onward.
                          Streamed with pyarrow and filtered by gameDate before any
                          pandas frame is built.
  * TeamHistories.csv     teamId, teamCity, teamName, teamAbbrev, seasonFounded,
                          seasonActiveTill, league. The NBA row active for the game's
                          season supplies the abbreviation.

Team resolution: `playerteamId`/`opponentteamId` are used when present. The dump
leaves them empty for all of 2021-22 and for a few later rows, so those rows are
resolved by (city, name) against TeamHistories, then by name alone when the city
spelling differs (e.g. "LA" vs "Los Angeles" Clippers). Rows resolved by name are
counted per season.

NBA Cup: group and knockout games are kept (they count as regular-season games);
the Cup final is excluded. The final is recognised by gameSubLabel "Championship"
or a game id with season-type code 006 (e.g. 0062300001), whichever the dump used.

Did-not-play rows: a row is treated as DNP when `numMinutes` is null or 0, or when
`comment` is populated (the dump uses it for "DNP - Coach's Decision", injury notes,
and similar). DNP rows are dropped and counted per season. `numMinutes` is parsed
from either a decimal ("39.1666") or "MM:SS".
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv

from nba import config, schema
from nba.storage import local

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
# PlayerStatistics.csv columns used to derive canonical columns, detect DNPs, or
# resolve teams when the id columns are empty.
BOX_SCORE_DERIVED: tuple[str, ...] = (
    "firstName",
    "lastName",
    "gameType",
    "gameLabel",
    "gameSubLabel",
    "comment",
    "playerteamCity",
    "playerteamName",
    "opponentteamCity",
    "opponentteamName",
)
TEAM_HISTORY_COLUMNS: tuple[str, ...] = (
    "teamId",
    "teamCity",
    "teamName",
    "teamAbbrev",
    "seasonFounded",
    "seasonActiveTill",
)
TEAM_HISTORY_LEAGUE_COLUMN = "league"
TEAM_HISTORY_LEAGUE = "NBA"

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

# Regular-season games listed in the dump's schedule files but absent from
# PlayerStatistics.csv (postponed and never re-captured upstream). Reported in the
# dataset card; to be backfilled from nba_api.
KNOWN_MISSING_GAMES: tuple[dict[str, str], ...] = (
    {
        "season": "2024-25",
        "game_id": "0022400524",
        "scheduled": "2025-01-09",
        "home": "LAL",
        "away": "CHA",
    },
    {
        "season": "2024-25",
        "game_id": "0022400532",
        "scheduled": "2025-01-11",
        "home": "ATL",
        "away": "HOU",
    },
    {
        "season": "2024-25",
        "game_id": "0022400537",
        "scheduled": "2025-01-11",
        "home": "LAL",
        "away": "SAN",
    },
    {
        "season": "2024-25",
        "game_id": "0022400538",
        "scheduled": "2025-01-11",
        "home": "LAC",
        "away": "CHA",
    },
    {
        "season": "2024-25",
        "game_id": "0022400617",
        "scheduled": "2025-01-22",
        "home": "NOP",
        "away": "MIL",
    },
    {
        "season": "2024-25",
        "game_id": "0022400627",
        "scheduled": "2025-01-23",
        "home": "UTA",
        "away": "WAS",
    },
    {
        "season": "2024-25",
        "game_id": "0022400988",
        "scheduled": "2025-03-17",
        "home": "SAN",
        "away": "ORL",
    },
)

# NBA Cup (in-season tournament) markers in the dump.
CUP_LABEL = "Emirates NBA Cup"  # gameLabel on every Cup game, whatever its gameType
CUP_FINAL_SUBLABEL = "Championship"  # gameSubLabel on the final (2024-25, 2025-26)
CUP_FINAL_GAME_ID_PREFIX = "006"  # season-type code of the final's game id (2023-24 onward)

# First game date kept when streaming the box-score file: October of the first season.
BACKFILL_START = pd.Timestamp(f"{config.SEASONS[0][:4]}-10-01")

# Format of gameDate in the dump ("2026-06-13 20:30:00"; empty for some non-NBA rows).
GAME_DATE_FORMAT = "ISO8601"

# pyarrow types for the streamed read. Text columns that need parsing stay strings;
# stats are float64 because DNP rows are empty.
_STRING_COLUMNS: tuple[str, ...] = (
    "gameId",
    "gameDate",
    "numMinutes",
    "firstName",
    "lastName",
    "gameType",
    "gameLabel",
    "gameSubLabel",
    "comment",
    "playerteamCity",
    "playerteamName",
    "opponentteamCity",
    "opponentteamName",
)
# Ids are read as float64 too: the team id columns are empty for many rows and
# may be written as "1610612747.0"; schema.coerce casts player_id to int64 later.
_BOX_SCORE_TYPES: dict[str, pa.DataType] = {
    **{c: pa.string() for c in _STRING_COLUMNS},
    **{c: pa.float64() for c in BOX_SCORE_COLUMNS if c not in _STRING_COLUMNS},
}


@dataclass(frozen=True)
class Prepared:
    """Output of `prepare`: the validated game logs plus per-season bookkeeping."""

    game_logs: pd.DataFrame
    dnp_per_season: pd.Series  # season -> DNP rows dropped
    name_resolved_per_season: pd.Series  # season -> kept rows whose team came from name lookup
    cup_games_per_season: pd.Series  # season -> distinct NBA Cup games kept (group + knockout)
    cup_final_per_season: pd.Series  # season -> distinct NBA Cup finals dropped
    game_type_counts: pd.Series  # gameType -> rows (before filtering)


class ThresholdError(ValueError):
    """A season is below the hard minimum row or game count."""


def check_thresholds(
    summary: pd.DataFrame,
    min_rows: int | None = None,
    min_games: int | None = None,
    full_games: int | None = None,
) -> list[str]:
    """Raise ThresholdError below the hard minimums; return warnings for incomplete seasons."""
    min_rows = config.MIN_ROWS_PER_SEASON if min_rows is None else min_rows
    min_games = config.MIN_GAMES_PER_SEASON if min_games is None else min_games
    full_games = config.FULL_SEASON_GAMES if full_games is None else full_games
    hard: list[str] = []
    warnings: list[str] = []
    for season, r in summary.iterrows():
        rows, games = int(r["rows"]), int(r["games"])
        if rows < min_rows:
            hard.append(f"{season}: {rows} rows < {min_rows}")
        if games < min_games:
            hard.append(f"{season}: {games} games < {min_games}")
        elif games != full_games:
            warnings.append(f"{season}: {games} games, expected {full_games}")
    if hard:
        raise ThresholdError("backfill below hard thresholds: " + "; ".join(hard))
    return warnings


def season_from_date(ts: pd.Timestamp) -> str:
    """Season label for a game date: start year is the year if month >= 10, else year - 1."""
    start_year = ts.year if ts.month >= 10 else ts.year - 1
    return f"{start_year}-{(start_year + 1) % 100:02d}"


def season_start_year(season: str) -> int:
    return int(season[:4])


def normalize_game_id(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip().str.zfill(GAME_ID_WIDTH)


def parse_minutes(s: pd.Series) -> pd.Series:
    """Minutes as float from either "39.1666" or "MM:SS". Empty/null -> NaN."""
    text = s.astype("string").str.strip()
    text = text.mask(text == "", pd.NA)
    clock = text.str.contains(":", na=False)
    parts = text.where(clock).str.split(":", expand=True, n=1)
    from_clock = pd.Series(pd.NA, index=s.index, dtype="Float64")
    if clock.any():
        mm = pd.to_numeric(parts[0], errors="coerce")
        ss = pd.to_numeric(parts[1], errors="coerce")
        from_clock = mm + ss / 60.0
    from_decimal = pd.to_numeric(text.where(~clock), errors="coerce")
    return from_clock.where(clock, from_decimal).astype("float64")


def _require(columns: Iterable[str], needed: Iterable[str], what: str) -> None:
    have = set(columns)
    missing = [c for c in needed if c not in have]
    if missing:
        raise KeyError(f"{what} is missing expected columns: {missing}")


def load_box_scores(
    kaggle_dir: Path, start: pd.Timestamp = BACKFILL_START, allow_empty: bool = False
) -> pd.DataFrame:
    """Stream PlayerStatistics.csv and keep only rows with gameDate >= start.

    `gameDate` is returned as datetime64 and `numMinutes` as float64 (parsed from
    decimal or "MM:SS"). Nothing is dropped here except pre-cutoff rows. With
    `allow_empty` an empty (typed) frame is returned instead of raising when no row
    is on or after `start`, which the daily ingest expects during the off-season.
    """
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
        dates = pd.to_datetime(df["gameDate"], format=GAME_DATE_FORMAT, errors="coerce")
        keep = dates >= start
        if keep.any():
            df = df[keep].copy()
            df["gameDate"] = dates[keep].dt.tz_localize(None)
            df["numMinutes"] = parse_minutes(df["numMinutes"])
            parts.append(df)
    if not parts:
        if not allow_empty:
            raise ValueError(f"no rows in {path} with gameDate >= {start.date()}")
        empty = reader.schema.empty_table().to_pandas()
        empty["gameDate"] = pd.to_datetime(empty["gameDate"])
        empty["numMinutes"] = empty["numMinutes"].astype("float64")
        return empty
    return pd.concat(parts, ignore_index=True)


def load_team_histories(kaggle_dir: Path) -> pd.DataFrame:
    """NBA rows of TeamHistories.csv with stripped text and an open-ended end season."""
    df = pd.read_csv(kaggle_dir / TEAM_HISTORY_FILE)
    _require(df.columns, TEAM_HISTORY_COLUMNS, TEAM_HISTORY_FILE)
    if TEAM_HISTORY_LEAGUE_COLUMN in df.columns:
        league = df[TEAM_HISTORY_LEAGUE_COLUMN].astype("string").str.strip()
        df = df[league == TEAM_HISTORY_LEAGUE]
    out = df[list(TEAM_HISTORY_COLUMNS)].copy()
    out["teamId"] = pd.to_numeric(out["teamId"]).astype("int64")
    for col in ("teamCity", "teamName", "teamAbbrev"):
        out[col] = out[col].astype("string").str.strip()
    out["seasonFounded"] = pd.to_numeric(out["seasonFounded"]).astype("int64")
    # Open-ended histories have no end season (the dump also uses 2100).
    out["seasonActiveTill"] = pd.to_numeric(out["seasonActiveTill"]).fillna(9999).astype("int64")
    return out.reset_index(drop=True)


def _lookup(keys: pd.DataFrame, histories: pd.DataFrame, on: list[str], label: str) -> pd.Series:
    """Abbreviation per row of `keys` (columns `on` + `year`) from the active history row.

    Returns NaN where no active row matches. Raises when more than one row matches.
    """
    unique = keys.drop_duplicates().reset_index(drop=True)
    merged = unique.merge(histories, on=on, how="left")
    is_active = (merged["seasonFounded"] <= merged["year"]) & (
        merged["year"] <= merged["seasonActiveTill"]
    )
    active = merged[is_active]
    counts = active.groupby(on + ["year"]).size().rename("n").reset_index()
    ambiguous = counts[counts["n"] > 1][on + ["year"]].to_dict("records")
    if ambiguous:
        raise KeyError(f"multiple active {TEAM_HISTORY_FILE} rows by {label} for: {ambiguous}")
    lookup = active[on + ["year", "teamAbbrev"]]
    abbrev = keys.merge(lookup, on=on + ["year"], how="left")["teamAbbrev"]
    return abbrev.astype("string").set_axis(keys.index)


def resolve_teams(
    team_ids: pd.Series,
    cities: pd.Series,
    names: pd.Series,
    season_years: pd.Series,
    histories: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    """Abbreviation for each row, plus a boolean marking rows resolved without an id.

    Order: team id, then (city, name), then name alone. Raises if any row stays unresolved.
    """
    years = season_years.astype("int64")
    ids = pd.to_numeric(team_ids, errors="coerce")
    city = cities.astype("string").str.strip()
    name = names.astype("string").str.strip()

    by_id = pd.DataFrame({"teamId": ids.fillna(-1).astype("int64"), "year": years})
    abbrev = _lookup(by_id, histories, ["teamId"], "id")
    abbrev = abbrev.mask(ids.isna(), pd.NA)

    missing = abbrev.isna()
    if missing.any():
        by_city_name = pd.DataFrame({"teamCity": city, "teamName": name, "year": years})[missing]
        abbrev.loc[missing] = _lookup(
            by_city_name, histories, ["teamCity", "teamName"], "city+name"
        )
    still_missing = abbrev.isna()
    if still_missing.any():
        by_name = pd.DataFrame({"teamName": name, "year": years})[still_missing]
        abbrev.loc[still_missing] = _lookup(by_name, histories, ["teamName"], "name")

    unresolved = abbrev.isna()
    if unresolved.any():
        sample = (
            pd.DataFrame({"teamId": ids, "city": city, "name": name, "year": years})[unresolved]
            .drop_duplicates()
            .head(10)
            .to_dict("records")
        )
        raise KeyError(f"could not resolve teams for: {sample}")
    return abbrev, missing & ~unresolved


def game_type_counts(box: pd.DataFrame) -> pd.Series:
    """Distinct gameType values with row counts, most common first."""
    return box["gameType"].astype("string").str.strip().value_counts(dropna=False)


def is_cup_game(game_label: pd.Series) -> pd.Series:
    """True for any NBA Cup game (group, knockout, or final)."""
    return (game_label.astype("string").str.strip() == CUP_LABEL).fillna(False)


def is_cup_final(game_id: pd.Series, game_sub_label: pd.Series) -> pd.Series:
    """True for the NBA Cup championship game, which does not count as a regular-season game."""
    by_sublabel = game_sub_label.astype("string").str.strip() == CUP_FINAL_SUBLABEL
    by_id = game_id.astype("string").str.startswith(CUP_FINAL_GAME_ID_PREFIX)
    return by_sublabel.fillna(False) | by_id.fillna(False)


def is_dnp(minutes: pd.Series, comment: pd.Series) -> pd.Series:
    """True for rows that did not play: no minutes, zero minutes, or a populated comment."""
    has_comment = comment.astype("string").str.strip().fillna("") != ""
    return minutes.isna() | (minutes <= 0) | has_comment


def prepare(
    box: pd.DataFrame,
    histories: pd.DataFrame,
    seasons: tuple[str, ...] | None = config.SEASONS,
    game_types: tuple[str, ...] = config.GAME_TYPES,
    require_primary_label: bool = True,
) -> Prepared:
    """Filter, drop DNPs, map to the canonical schema, validate, and return bookkeeping.

    `seasons=None` keeps every season present, and `require_primary_label=False`
    accepts a frame without any "Regular Season" row; both are used by the daily
    ingest, whose window can be a handful of days.
    """
    _require(box.columns, list(BOX_SCORE_COLUMNS) + list(BOX_SCORE_DERIVED), BOX_SCORE_FILE)

    type_counts = game_type_counts(box)
    present = set(type_counts.index.dropna())
    # The primary label must exist; the Cup labels vary by season and may be absent.
    if require_primary_label and game_types[0] not in present:
        raise ValueError(
            f"gameType value {game_types[0]!r} not found; present values: {sorted(present)}"
        )

    kept = box["gameType"].astype("string").str.strip().isin(game_types)
    df = box[kept].rename(columns=BOX_SCORE_COLUMNS).copy()
    df["game_id"] = normalize_game_id(df["game_id"])
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.tz_localize(None).dt.normalize()
    df["season"] = df["game_date"].map(season_from_date)
    if seasons is not None:
        df = df[df["season"].isin(seasons)]
    season_index = sorted(df["season"].unique())

    final = is_cup_final(df["game_id"], df["gameSubLabel"])
    cup_final = df.loc[final].groupby("season")["game_id"].nunique()
    cup_final = cup_final.reindex(season_index, fill_value=0)
    df = df[~final]
    cup = is_cup_game(df["gameLabel"])
    cup_games = df.loc[cup].groupby("season")["game_id"].nunique()
    cup_games = cup_games.reindex(season_index, fill_value=0)

    dnp = is_dnp(df["minutes"], df["comment"])
    dnp_per_season = df.loc[dnp].groupby("season").size().reindex(season_index, fill_value=0)
    df = df[~dnp].copy()

    first = df["firstName"].astype("string").str.strip()
    last = df["lastName"].astype("string").str.strip()
    df["player_name"] = first + " " + last
    years = df["season"].map(season_start_year)
    df["team"], by_name_team = resolve_teams(
        df["team_id"], df["playerteamCity"], df["playerteamName"], years, histories
    )
    df["opponent"], by_name_opp = resolve_teams(
        df["opponent_id"], df["opponentteamCity"], df["opponentteamName"], years, histories
    )
    by_name = by_name_team | by_name_opp
    name_resolved = df.loc[by_name].groupby("season").size().reindex(season_index, fill_value=0)

    df["home"] = pd.to_numeric(df["home"]).astype("int64").astype("bool")
    for col in COUNTING_STATS:
        df[col] = pd.to_numeric(df[col]).fillna(0).round().astype("int64")
    df["source"] = config.KAGGLE_SOURCE

    out = schema.coerce(df)
    out = out.sort_values(["game_date", "game_id", "player_id"]).reset_index(drop=True)
    return Prepared(
        schema.validate(out),
        dnp_per_season.rename("dnp_dropped"),
        name_resolved.rename("team_by_name"),
        cup_games.rename("cup_games"),
        cup_final.rename("cup_final_dropped"),
        type_counts,
    )


def map_to_schema(
    box: pd.DataFrame,
    histories: pd.DataFrame,
    seasons: tuple[str, ...] | None = config.SEASONS,
    game_types: tuple[str, ...] = config.GAME_TYPES,
) -> pd.DataFrame:
    """Map the raw box-score frame to the canonical schema and validate."""
    return prepare(box, histories, seasons, game_types).game_logs


def backfill(kaggle_dir: Path, out_dir: Path) -> dict[str, Path]:
    df = map_to_schema(load_box_scores(kaggle_dir), load_team_histories(kaggle_dir))
    return local.write_per_season(df, out_dir)


def season_summary(prepared: Prepared) -> pd.DataFrame:
    """Per season: rows, games, first/last date, DNP dropped, name-resolved, Cup games/finals."""
    df = prepared.game_logs
    summary = df.groupby("season").agg(
        rows=("game_id", "size"),
        games=("game_id", "nunique"),
        first_game=("game_date", "min"),
        last_game=("game_date", "max"),
    )
    extras = [
        prepared.dnp_per_season,
        prepared.name_resolved_per_season,
        prepared.cup_games_per_season,
        prepared.cup_final_per_season,
    ]
    for extra in extras:
        summary = summary.join(extra)
    counts = {str(e.name): "int64" for e in extras}
    return summary.fillna(dict.fromkeys(counts, 0)).astype(counts)
