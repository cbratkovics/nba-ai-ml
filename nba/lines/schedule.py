"""The schedule fallback (ADR-0029): The Odds API's free `/events` endpoint stands in for the
Kaggle dump's LeagueSchedule file when that file does not exist, and its event ids are mapped
to NBA game ids once the box scores arrive.

  events_to_games       events -> one row per game with the Eastern-time game date, both
                        teams as the game logs' abbreviations, and a provisional game id
                        `evt:<event id>` that can never collide with an nba.com id
  fallback_schedule     the same rows in the layout nba.ingest.schedule.games_on expects, so
                        the slate runs unchanged (schedule_source = odds_api)
  resolve_game_ids      provisional ids -> real game ids by (game date, team) against the
                        stored game logs; rows that stay provisional are counted, never dropped
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from nba import config
from nba.ingest.kaggle_dump import season_from_date
from nba.lines.teams import abbreviation

EASTERN = ZoneInfo("America/New_York")
PROVISIONAL_PREFIX = "evt:"
SOURCE_ODDS_API = "odds_api"
SOURCE_KAGGLE = "kaggle_dump"


def eastern_date(commence_time: str) -> date:
    t = datetime.fromisoformat(commence_time.replace("Z", "+00:00"))
    return t.astimezone(EASTERN).date()


def in_regular_season(d: date) -> bool:
    """True when the Eastern date falls inside its season's configured regular-season window
    (config.REGULAR_SEASON_WINDOWS); an unknown season or a date before opening night is out.
    An open-ended window (end None) accepts every date from opening night on."""
    window = config.REGULAR_SEASON_WINDOWS.get(season_from_date(pd.Timestamp(d)))
    if window is None:
        return False
    start, end = window
    if d < date.fromisoformat(start):
        return False
    return end is None or d <= date.fromisoformat(end)


def classify_event(event: dict[str, Any]) -> str:
    """regular_season, or why the event is out of season: wrong sport key (preseason, All-Star,
    summer league are separate keys at The Odds API) or a date outside the window."""
    key = event.get("sport_key", config.ODDS_SPORT)
    if key not in config.ODDS_SPORT_KEYS_IN_SCOPE:
        return f"sport_key:{key}"
    if not in_regular_season(eastern_date(event["commence_time"])):
        return "outside_regular_season_window"
    return "regular_season"


def events_to_games(events: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for e in events:
        rows.append(
            {
                "event_id": str(e["id"]),
                "commence_time_utc": pd.Timestamp(e["commence_time"]).tz_convert("UTC"),
                "game_date": eastern_date(e["commence_time"]),
                "home": abbreviation(e["home_team"]),
                "away": abbreviation(e["away_team"]),
                "home_team_name": e["home_team"],
                "away_team_name": e["away_team"],
            }
        )
    cols = [
        "event_id",
        "commence_time_utc",
        "game_date",
        "home",
        "away",
        "home_team_name",
        "away_team_name",
    ]
    return pd.DataFrame(rows, columns=cols)


def provisional_game_id(event_id: str) -> str:
    return f"{PROVISIONAL_PREFIX}{event_id}"


def is_provisional(game_id: pd.Series) -> pd.Series:
    return game_id.astype(str).str.startswith(PROVISIONAL_PREFIX)


def fallback_schedule(events: list[dict[str, Any]]) -> pd.DataFrame:
    """game_id, game_date, home, away, season, event_id, schedule_source, in the slate's layout.
    Only regular-season events (classify_event) become games; a preseason or play-in event
    never produces a slate."""
    events = [e for e in events if classify_event(e) == "regular_season"]
    g = events_to_games(events)
    out = pd.DataFrame(
        {
            "game_id": g["event_id"].map(provisional_game_id),
            "game_date": pd.to_datetime(g["game_date"]),
            "home": g["home"],
            "away": g["away"],
        }
    )
    out["season"] = out["game_date"].map(season_from_date)
    out["tip_off_utc"] = g["commence_time_utc"]
    out["event_id"] = g["event_id"].astype("string")
    out["schedule_source"] = SOURCE_ODDS_API
    return out.sort_values(["game_date", "game_id"]).reset_index(drop=True)


def resolve_game_ids(frame: pd.DataFrame, game_logs: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Replace provisional game ids with the real id of the game the team played on that date.

    `frame` needs game_id, game_date (date or timestamp) and team. A team plays at most once
    per date, so (game_date, team) identifies the game. Returns the frame and the number of
    rows that stayed provisional (no box score for that team on that date yet)."""
    out = frame.copy()
    prov = is_provisional(out["game_id"])
    if not prov.any():
        return out, 0
    key = (
        game_logs[["game_id", "game_date", "team"]]
        .drop_duplicates()
        .assign(game_date=lambda d: pd.to_datetime(d["game_date"]).dt.normalize())
        .rename(columns={"game_id": "resolved_game_id"})
    )
    sub = out.loc[prov, ["game_date", "team"]].copy()
    sub["game_date"] = pd.to_datetime(sub["game_date"]).dt.normalize()
    merged = sub.merge(key, on=["game_date", "team"], how="left")
    resolved = merged["resolved_game_id"].to_numpy()
    found = pd.notna(resolved)
    idx = out.index[prov]
    out.loc[idx[found], "game_id"] = resolved[found]
    return out, int((~found).sum())
