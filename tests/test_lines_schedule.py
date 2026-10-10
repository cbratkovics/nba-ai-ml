"""The schedule fallback (ADR-0029): events become a slate schedule with provisional ids, and
the ids resolve to real game ids by (date, team) once the box scores exist."""

from __future__ import annotations

from datetime import date

import pandas as pd

from nba.ingest import schedule as kaggle_schedule
from nba.lines import schedule

EVENTS = [
    # 7:30 pm ET on Oct 20 is 23:30 UTC; 10:00 pm ET is 02:00 UTC the next day: same ET date.
    {
        "id": "abc123",
        "commence_time": "2026-10-20T23:30:00Z",
        "home_team": "Boston Celtics",
        "away_team": "New York Knicks",
    },
    {
        "id": "def456",
        "commence_time": "2026-10-21T02:00:00Z",
        "home_team": "Los Angeles Lakers",
        "away_team": "San Antonio Spurs",
    },
]


def test_events_become_games_on_the_eastern_date() -> None:
    g = schedule.events_to_games(EVENTS)
    assert g["game_date"].tolist() == [date(2026, 10, 20), date(2026, 10, 20)]
    assert g["home"].tolist() == ["BOS", "LAL"] and g["away"].tolist() == ["NYK", "SAN"]


def test_fallback_schedule_feeds_the_slate_unchanged() -> None:
    sched = schedule.fallback_schedule(EVENTS)
    assert set(sched["season"]) == {"2026-27"} and set(sched["schedule_source"]) == {"odds_api"}
    assert sched["game_id"].tolist() == ["evt:abc123", "evt:def456"]
    on = kaggle_schedule.games_on(sched, date(2026, 10, 20))
    assert len(on) == 2 and kaggle_schedule.games_on(sched, date(2026, 10, 21)).empty
    assert schedule.is_provisional(sched["game_id"]).all()


def test_provisional_ids_resolve_by_date_and_team_and_count_the_rest() -> None:
    preds = pd.DataFrame(
        {
            "game_id": ["evt:abc123", "evt:abc123", "evt:def456", "0022600099"],
            "game_date": pd.to_datetime(["2026-10-20"] * 4),
            "team": ["BOS", "NYK", "LAL", "MIA"],
            "player_id": [1, 2, 3, 4],
        }
    )
    logs = pd.DataFrame(
        {
            "game_id": ["0022600001", "0022600001"],
            "game_date": pd.to_datetime(["2026-10-20", "2026-10-20"]),
            "team": ["BOS", "NYK"],
        }
    )
    out, unmatched = schedule.resolve_game_ids(preds, logs)
    assert out["game_id"].tolist() == ["0022600001", "0022600001", "evt:def456", "0022600099"]
    assert unmatched == 1
    again, n = schedule.resolve_game_ids(out, logs)
    assert n == 1 and again.equals(out)  # idempotent; the LAL game is still not ingested
    none, n0 = schedule.resolve_game_ids(preds[preds["game_id"] == "0022600099"], logs)
    assert n0 == 0


def test_regular_season_window_and_fallback_drop_preseason() -> None:
    assert not schedule.in_regular_season(date(2026, 10, 15))  # preseason
    assert schedule.in_regular_season(date(2026, 10, 20))  # opening night
    assert schedule.in_regular_season(date(2027, 3, 1))  # open-ended window until the end is set
    assert schedule.in_regular_season(date(2026, 4, 12)) and not schedule.in_regular_season(
        date(2026, 4, 13)
    )
    assert not schedule.in_regular_season(date(2030, 1, 1))  # no window configured
    pre = [
        dict(e, commence_time=e["commence_time"].replace("2026-10-2", "2026-10-1")) for e in EVENTS
    ]  # 2026-10-10 and 2026-10-11: preseason dates
    assert schedule.fallback_schedule(pre).empty
    tagged = [dict(EVENTS[0], sport_key="basketball_nba_preseason"), EVENTS[1]]
    assert schedule.fallback_schedule(tagged)["game_id"].tolist() == ["evt:def456"]
    assert schedule.classify_event(
        dict(EVENTS[0], sport_key="basketball_nba_all_stars")
    ).startswith("sport_key:")
