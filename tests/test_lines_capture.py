"""The capture (ADR-0031): the decision record precedes every priced call, games under way are
out of scope and counted, the guard refuses an over-budget plan before any fetch, lines are
flattened to over/under pairs and matched deterministically, and nothing retries."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import pytest

from nba import config
from nba.lines import capture, store
from nba.lines.odds_api import Answer, OddsApiError, Usage
from nba.storage import local

D = date(2026, 10, 20)
PUSHES: list[list[str]] = []  # the files each store.push call received, per test (see root)
SNAPSHOT = datetime(2026, 10, 20, 20, 0, tzinfo=UTC)


def _usage(last: int, remaining: int = 400) -> Usage:
    return Usage(remaining=remaining, used=500 - remaining, last=last, at=SNAPSHOT.isoformat())


class FakeApi:
    def __init__(self, events, odds_by_event, fail: set[str] = frozenset()):
        self._events, self._odds, self._fail = events, odds_by_event, set(fail)
        self.calls: list[str] = []

    def events(self, commence_from=None, commence_to=None):
        self.calls.append("events")
        return Answer(self._events, _usage(0), "/sports/basketball_nba/events")

    def event_odds(self, event_id, markets=config.ODDS_MARKETS, bookmakers=config.ODDS_BOOKMAKERS):
        self.calls.append(f"odds:{event_id}")
        if event_id in self._fail:
            raise OddsApiError("/odds: HTTP 500: boom")
        return Answer(
            self._odds[event_id], _usage(1), f"/sports/basketball_nba/events/{event_id}/odds"
        )


@pytest.fixture
def root(tmp_path: Path, game_logs: pd.DataFrame, monkeypatch: pytest.MonkeyPatch) -> Path:
    r = tmp_path / "root"
    local.write_per_season(game_logs, r / config.DATA_DIR)
    PUSHES.clear()
    monkeypatch.setattr(store, "ensure_private_repo", lambda: None)
    monkeypatch.setattr(store, "pull", lambda root: "sha-pulled")
    monkeypatch.setattr(
        store, "push", lambda root, files, message: PUSHES.append([p.name for p in files]) or "sha"
    )
    return r


def _two_players(game_logs: pd.DataFrame, team: str) -> list[str]:
    return sorted(game_logs.loc[game_logs["team"] == team, "player_name"].unique())[:2]


def _events_and_odds(game_logs: pd.DataFrame):
    lal, bos = _two_players(game_logs, "LAL"), _two_players(game_logs, "BOS")
    events = [
        {
            "id": "late",
            "commence_time": "2026-10-21T00:00:00Z",
            "home_team": "Los Angeles Lakers",
            "away_team": "Boston Celtics",
        },
        {
            "id": "early",
            "commence_time": "2026-10-20T19:00:00Z",
            "home_team": "Miami Heat",
            "away_team": "Denver Nuggets",
        },
        {
            "id": "tomorrow",
            "commence_time": "2026-10-21T23:00:00Z",
            "home_team": "Miami Heat",
            "away_team": "Denver Nuggets",
        },
    ]
    outcomes = []
    for name, point, over, under in (
        (lal[0], 24.5, -115, -105),
        (bos[0], 18.5, -110, -110),
        ("Nobody Known", 9.5, 100, -120),
    ):
        outcomes.append({"name": "Over", "description": name, "price": over, "point": point})
        outcomes.append({"name": "Under", "description": name, "price": under, "point": point})
    odds = {
        "late": {
            "id": "late",
            "bookmakers": [
                {
                    "key": "draftkings",
                    "title": "DraftKings",
                    "markets": [
                        {
                            "key": "player_points",
                            "last_update": "2026-10-20T19:50:00Z",
                            "outcomes": outcomes,
                        }
                    ],
                }
            ],
        }
    }
    return events, odds, lal[0], bos[0]


def test_capture_records_the_decision_first_and_flattens_matched_lines(
    root: Path, game_logs: pd.DataFrame
) -> None:
    events, odds, lal_name, bos_name = _events_and_odds(game_logs)
    api = FakeApi(events, odds)
    cap = capture.run(D, root, trigger="schedule", pull=True, api=api, now=SNAPSHOT)
    # Decision pushed before any priced call, then the final push with lines + usage.
    assert PUSHES[0] == ["2026-10-20.json"]
    assert api.calls == [
        "events",
        "odds:late",
    ]  # the 19:00 UTC game is out of scope; tomorrow is not today
    assert (
        cap.status == "captured"
        and cap.n_in_scope == 1
        and cap.n_out_of_scope == 1
        and cap.n_events == 2
    )
    assert cap.planned_cost == 1 and cap.charged == 1 and cap.n_calls_ok == 1
    lines = pd.read_parquet(root / config.LINES_DIR / "2026-10-20.parquet")
    assert list(lines.columns) == list(capture.LINE_COLUMNS) and len(lines) == 3
    row = lines[lines["player_name_raw"] == lal_name].iloc[0]
    assert row["over_price"] == -115 and row["under_price"] == -105 and row["line"] == 24.5
    assert row["team"] == "LAL" and row["match_method"] == "team" and row["home_team"] == "LAL"
    assert lines[lines["player_name_raw"] == bos_name]["team"].iloc[0] == "BOS"
    assert cap.n_unmatched == 1 and cap.n_lines == 3 and cap.n_players == 3
    assert sorted(PUSHES[-1]) == ["2026-10-20.json", "2026-10-20.parquet", "odds_api.jsonl"]  # type: ignore[attr-defined]
    usage = [json.loads(ln) for ln in (root / config.ODDS_USAGE_FILE).read_text().splitlines()]
    assert [u["outcome"] for u in usage] == ["ok", "ok"] and usage[1]["charged"] == 1
    saved = json.loads((root / config.CAPTURES_DIR / "2026-10-20.json").read_text())
    assert saved["decided_before_fetch"] is True and saved["status"] == "captured"


def test_guard_refusal_happens_before_any_priced_call(root: Path, game_logs: pd.DataFrame) -> None:
    events, odds, _, _ = _events_and_odds(game_logs)
    usage = root / config.ODDS_USAGE_FILE
    usage.parent.mkdir(parents=True)
    usage.write_text(
        json.dumps(
            {
                "at": "2026-10-19T20:00:00+00:00",
                "endpoint": "x",
                "event_id": "e",
                "planned_cost": 375,
                "charged": 375,
                "remaining": 125,
                "used": 375,
                "outcome": "ok",
                "note": "",
                "month": "2026-10",
            }
        )
        + "\n"
    )
    api = FakeApi(events, odds)
    cap = capture.run(D, root, trigger="schedule", pull=False, api=api, now=SNAPSHOT)
    assert cap.status == "refused" and api.calls == ["events"]
    assert "ceiling" in cap.notes[0]
    assert not (root / config.LINES_DIR / "2026-10-20.parquet").exists()
    entries = [json.loads(ln) for ln in usage.read_text().splitlines()]
    assert entries[-1]["outcome"] == "refused" and entries[-1]["charged"] is None


def test_failed_priced_call_is_logged_once_and_never_retried(
    root: Path, game_logs: pd.DataFrame
) -> None:
    events, odds, _, _ = _events_and_odds(game_logs)
    api = FakeApi(events, odds, fail={"late"})
    cap = capture.run(D, root, trigger="schedule", pull=False, api=api, now=SNAPSHOT)
    assert api.calls == ["events", "odds:late"] and cap.n_calls_failed == 1 and cap.n_lines == 0
    assert cap.status == "captured"  # the night is on record even with zero lines
    entries = [json.loads(ln) for ln in (root / config.ODDS_USAGE_FILE).read_text().splitlines()]
    assert entries[-1]["outcome"] == "error"


def test_no_in_scope_games_and_already_captured(root: Path, game_logs: pd.DataFrame) -> None:
    events, odds, _, _ = _events_and_odds(game_logs)
    api = FakeApi([events[1]], odds)  # only the game already under way
    cap = capture.run(D, root, trigger="schedule", pull=False, api=api, now=SNAPSHOT)
    assert cap.status == "no_games" and cap.n_out_of_scope == 1 and api.calls == ["events"]
    # A second run on a captured night does nothing.
    api2 = FakeApi(events, odds)
    capture.run(D, root, trigger="dispatch", pull=False, api=api2, now=SNAPSHOT)
    (root / config.CAPTURES_DIR / "2026-10-20.json").write_text(
        json.dumps(
            {
                **json.loads((root / config.CAPTURES_DIR / "2026-10-20.json").read_text()),
                "status": "captured",
            }
        )
    )
    api3 = FakeApi(events, odds)
    again = capture.run(D, root, trigger="dispatch", pull=False, api=api3, now=SNAPSHOT)
    assert again.status == "captured" and api3.calls == []


def test_dispatch_snapshot_is_now_and_plan_cost_follows_the_rule() -> None:
    assert capture.snapshot_time(D) == SNAPSHOT
    events = [
        {"id": "a", "commence_time": "2026-10-20T23:00:00Z"},
        {"id": "b", "commence_time": "2026-10-20T19:30:00Z"},
        {"id": "c", "commence_time": "2026-10-21T02:30:00Z"},  # 22:30 ET, same ET date
    ]
    plan, out = capture.plan_events(events, D, SNAPSHOT)
    assert [e["id"] for e in plan] == ["a", "c"] and out == 1
    start, end = capture.eastern_day_bounds_utc(D)
    assert start == "2026-10-20T04:00:00Z" and end == "2026-10-21T04:00:00Z"
