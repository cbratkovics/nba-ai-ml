"""League schedule files from the Kaggle dump (LeagueScheduleYY_YY.csv).

The two files in the dump differ in column names (`hometeamId` in 24_25,
`homeTeamId` in 25_26); both are accepted. Game dates are the Eastern-time
calendar date of `gameDateTimeEst`, which is the NBA's own game date.

The file for a date's season is the only one consulted for slates. A missing file
means "no schedule for that season" (the author has not published it yet), which
is reported separately from "schedule exists but no games on that date".
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from nba.ingest.kaggle_dump import (
    GAME_ID_WIDTH,
    is_cup_final,
    normalize_game_id,
    resolve_teams,
    season_from_date,
    season_start_year,
)

COLUMN_ALIASES: dict[str, str] = {
    "gameId": "game_id",
    "gameDateTimeEst": "game_datetime_est",
    "hometeamId": "home_team_id",
    "homeTeamId": "home_team_id",
    "awayteamId": "away_team_id",
    "awayTeamId": "away_team_id",
    "gameLabel": "game_label",
    "gameSubLabel": "game_sub_label",
}
REQUIRED: tuple[str, ...] = ("game_id", "game_datetime_est", "home_team_id", "away_team_id")
REGULAR_SEASON_PREFIX = "002"


def schedule_file_name(season: str) -> str:
    """'2025-26' -> 'LeagueSchedule25_26.csv'."""
    start = season_start_year(season)
    return f"LeagueSchedule{start % 100:02d}_{(start + 1) % 100:02d}.csv"


def season_for_date(d: date) -> str:
    return season_from_date(pd.Timestamp(d))


def schedule_path(directory: Path, d: date) -> Path | None:
    """Path of the schedule file for the date's season, or None if it does not exist."""
    path = directory / schedule_file_name(season_for_date(d))
    return path if path.exists() else None


def load_schedule(path: Path, histories: pd.DataFrame) -> pd.DataFrame:
    """Regular-season games with abbreviations: game_id, game_date, home, away, season."""
    raw = pd.read_csv(path, low_memory=False)
    df = raw.rename(columns=COLUMN_ALIASES)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise KeyError(f"{path.name} is missing expected columns: {missing}")
    for col in ("game_label", "game_sub_label"):
        if col not in df.columns:
            df[col] = pd.NA

    out = pd.DataFrame(
        {
            "game_id": normalize_game_id(df["game_id"]),
            "game_datetime_est": pd.to_datetime(df["game_datetime_est"], format="ISO8601"),
            "home_team_id": pd.to_numeric(df["home_team_id"]),
            "away_team_id": pd.to_numeric(df["away_team_id"]),
            "game_label": df["game_label"].astype("string"),
            "game_sub_label": df["game_sub_label"].astype("string"),
        }
    )
    out["game_date"] = out["game_datetime_est"].dt.normalize()
    out["season"] = out["game_date"].map(season_from_date)
    regular = out["game_id"].str.startswith(REGULAR_SEASON_PREFIX) & ~is_cup_final(
        out["game_id"], out["game_sub_label"]
    )
    out = out[regular].copy()
    if out.empty:
        return out.assign(home=pd.Series(dtype="string"), away=pd.Series(dtype="string"))

    years = out["season"].map(season_start_year)
    blank = pd.Series([pd.NA] * len(out), index=out.index, dtype="string")
    out["home"], _ = resolve_teams(out["home_team_id"], blank, blank, years, histories)
    out["away"], _ = resolve_teams(out["away_team_id"], blank, blank, years, histories)
    assert out["game_id"].str.len().eq(GAME_ID_WIDTH).all()
    return out.sort_values(["game_date", "game_id"]).reset_index(drop=True)


def games_on(schedule: pd.DataFrame, d: date) -> pd.DataFrame:
    """Regular-season games scheduled on the given Eastern-time date."""
    return schedule[schedule["game_date"] == pd.Timestamp(d)].reset_index(drop=True)
