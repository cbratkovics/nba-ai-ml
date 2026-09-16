"""One drift check: the window of games played before a date against the reference.

Window = training-population rows (minutes >= MIN_MINUTES, an earlier game this season)
with a game date in [d - DRIFT_WINDOW_DAYS, d - 1], features from the one feature module
over the stored game logs. Season position (opening, cup, deadline_week, all_star_return,
april, regular) is a label from the season's game dates and SEASON_CALENDAR; the reference
mode is `day_NNN`, the run date's season day (days since the season's first game date), so
the window is compared with exactly the same season days of the training seasons
(ADR-0017); `all` is the season-long reference, used only before a season's first game.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import pandas as pd

from nba import config
from nba.drift import psi
from nba.features import asof

POSITIONS: tuple[str, ...] = (
    "opening",
    "cup",
    "deadline_week",
    "all_star_return",
    "april",
    "regular",
)


def season_of(d: date) -> str:
    start = d.year if d.month >= 10 else d.year - 1
    return f"{start}-{str(start + 1)[-2:]}"


def population_rows(game_logs: pd.DataFrame) -> pd.DataFrame:
    """Feature rows with the population flag the gold marts use (ADR-0009)."""
    logs = game_logs.copy()
    logs["game_date"] = pd.to_datetime(logs["game_date"])
    feats = asof.build_features(logs)  # sorted by player, date, game; carries season
    prior_season = feats.groupby(["player_id", "season"]).cumcount()
    feats["in_population"] = (feats["minutes"] >= config.MIN_MINUTES) & (prior_season >= 1)
    return feats


def season_positions(game_dates: list[date], season: str) -> dict[date, str]:
    """Position label per game date of a season."""
    dates = sorted(set(game_dates))
    cal = config.SEASON_CALENDAR.get(season, {})
    labels: dict[date, str] = {}
    opening = set(dates[: config.OPENING_GAME_DATES])
    after_break: set[date] = set()
    if "all_star_break" in cal:
        _, end = (date.fromisoformat(x) for x in cal["all_star_break"])
        after_break = set([d for d in dates if d > end][: config.ALL_STAR_RETURN_GAME_DATES])
    for d in dates:
        if d in opening:
            labels[d] = "opening"
        elif "deadline_week" in cal and _within(d, cal["deadline_week"]):
            labels[d] = "deadline_week"
        elif d in after_break:
            labels[d] = "all_star_return"
        elif "cup" in cal and _within(d, cal["cup"]):
            labels[d] = "cup"
        elif d.month == 4:
            labels[d] = "april"
        else:
            labels[d] = "regular"
    return labels


def _within(d: date, span: tuple[str, str]) -> bool:
    lo, hi = (date.fromisoformat(x) for x in span)
    return lo <= d <= hi


def reference_mode(d: date, season_game_dates: list[date], reference: dict[str, Any]) -> str:
    """The season-day key of d (clipped to the last day the training seasons reached);
    `all` before the season's first game."""
    dates = sorted(set(season_game_dates))
    if not dates or d <= dates[0]:
        return "all"
    day = min((d - dates[0]).days, int(reference.get("max_season_day", 0)) + 1)
    return f"day_{day:03d}"


def expected_for(reference: dict[str, Any], mode: str) -> dict[str, list[float]]:
    block = reference["all"] if mode == "all" else reference["daily"][mode]
    return block["expected"]


def window_rows(feats: pd.DataFrame, d: date) -> pd.DataFrame:
    start = pd.Timestamp(d - timedelta(days=config.DRIFT_WINDOW_DAYS))
    end = pd.Timestamp(d - timedelta(days=1))
    dates = feats["game_date"]
    return feats[feats["in_population"] & (dates >= start) & (dates <= end)]


def psi_by_feature(window: pd.DataFrame, reference: dict[str, Any], mode: str) -> dict[str, float]:
    expected = expected_for(reference, mode)
    return {f: psi.psi(window[f], reference["bins"][f], expected[f]) for f in reference["features"]}


def check(
    feats: pd.DataFrame,
    d: date,
    reference: dict[str, Any],
    *,
    season_game_dates: list[date] | None = None,
) -> dict[str, Any]:
    """PSI per feature for the window ending the day before d; no verdict yet."""
    season = season_of(d)
    if season_game_dates is None:
        in_season = feats[feats["season"] == season]
        season_game_dates = sorted({t.date() for t in in_season["game_date"].unique()})
    window = window_rows(feats, d)
    positions = season_positions(season_game_dates, season)
    grace = timedelta(days=config.DRIFT_WINDOW_DAYS)
    off_season = not season_game_dates or not (
        min(season_game_dates) <= d <= max(season_game_dates) + grace
    )
    position = "off_season" if off_season else positions.get(d, "regular")
    mode = "all" if off_season else reference_mode(d, season_game_dates, reference)
    n = int(len(window))
    values = psi_by_feature(window, reference, mode) if n >= config.DRIFT_MIN_ROWS else {}
    return {
        "date": d.isoformat(),
        "season": season,
        "position": position,
        "reference_mode": mode,
        "window": {
            "start": (d - timedelta(days=config.DRIFT_WINDOW_DAYS)).isoformat(),
            "end": (d - timedelta(days=1)).isoformat(),
            "days": config.DRIFT_WINDOW_DAYS,
            "n_rows": n,
            "min_rows": config.DRIFT_MIN_ROWS,
        },
        "psi": values,
    }
