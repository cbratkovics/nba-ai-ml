"""The credit guard fails closed (ADR-0031): an over-budget call is refused before it is made,
the provider's own remaining-credits reading can refuse too, every decision is logged, and the
documented cost rule is what the guard is asked about."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from nba import config
from nba.lines import guard
from nba.lines.odds_api import Usage, event_odds_cost


def _usage(remaining: int, used: int, last: int, at: str = "2026-10-20T20:00:00+00:00") -> Usage:
    return Usage(remaining=remaining, used=used, last=last, at=at)


def test_cost_rule_matches_the_documentation() -> None:
    assert event_odds_cost(1, 1) == 1  # one market, one bookmaker
    assert event_odds_cost(3, 1) == 3  # three markets
    assert event_odds_cost(1, 10) == 1  # ten bookmakers are one region unit
    assert event_odds_cost(1, 11) == 2
    assert event_odds_cost(len(config.ODDS_MARKETS), len(config.ODDS_BOOKMAKERS)) == 1


def test_ceilings_follow_the_config() -> None:
    assert guard.budget_ceiling(use_reserve=True) == 425
    assert guard.budget_ceiling(use_reserve=False) == 375


def test_guard_blocks_a_call_that_would_cross_the_ceiling(tmp_path: Path) -> None:
    g = guard.CreditGuard(tmp_path / "usage.jsonl", now=datetime(2026, 10, 25, tzinfo=UTC))
    # 370 already spent this month: 5 more is fine, 6 is not; a manual run may use the reserve.
    g.record("/events/x/odds", "x", 370, "ok", _usage(130, 370, 370))
    assert g.allow(5).allowed
    v = g.allow(6)
    assert not v.allowed and "375 ceiling" in v.reason and v.spent_this_month == 370
    assert g.allow(6, use_reserve=True).allowed
    assert not g.allow(56, use_reserve=True).allowed
    assert g.allow(0).reason == "free call"


def test_provider_remaining_can_refuse_on_its_own(tmp_path: Path) -> None:
    """The provider's cycle is not the calendar month: if it says fewer credits are left than the
    margin keeps back, the smaller figure wins even when the log looks fine."""
    g = guard.CreditGuard(tmp_path / "usage.jsonl", now=datetime(2026, 11, 2, tzinfo=UTC))
    # Nothing spent this calendar month, but the provider reports 120 remaining.
    g.record("/events/y/odds", "y", 1, "ok", _usage(120, 380, 1, at="2026-10-31T20:00:00+00:00"))
    assert g.spent_this_month() == 0
    v = g.allow(1)
    assert not v.allowed and "provider reports 120 remaining" in v.reason
    # 500 - 375 = 125 must stay out of reach on a scheduled run; a manual one keeps 75 back.
    assert g.allow(1, use_reserve=True).allowed  # 120 - 1 = 119 >= 75
    g.record("/events/z/odds", "z", 1, "ok", _usage(76, 424, 1, at="2026-11-02T20:00:00+00:00"))
    assert not g.allow(2, use_reserve=True).allowed


def test_log_is_append_only_and_refusals_are_recorded(tmp_path: Path) -> None:
    path = tmp_path / "usage.jsonl"
    g = guard.CreditGuard(path, now=datetime(2026, 10, 20, tzinfo=UTC))
    g.record("/events", None, 0, "ok", _usage(425, 75, 0))
    g.record("/events/a/odds", "a", 1, "refused", None, note="would cross")
    g.record("/events/b/odds", "b", 1, "error", None, note="HTTP 500")
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert [e["outcome"] for e in lines] == ["ok", "refused", "error"]
    assert lines[1]["charged"] is None and lines[2]["charged"] is None
    reloaded = guard.CreditGuard(path, now=datetime(2026, 10, 20, tzinfo=UTC))
    assert reloaded.summary()["refusals_this_month"] == 1
    assert reloaded.summary()["calls_this_month"] == 3
    assert reloaded.provider_remaining() == 425


def test_a_month_boundary_resets_the_log_reading_but_not_the_provider_reading(
    tmp_path: Path,
) -> None:
    g = guard.CreditGuard(tmp_path / "usage.jsonl", now=datetime(2026, 10, 31, 23, tzinfo=UTC))
    g.record(
        "/events/a/odds", "a", 375, "ok", _usage(125, 375, 375, at="2026-10-31T20:00:00+00:00")
    )
    assert not g.allow(1).allowed
    later = guard.CreditGuard(g.log_path, now=datetime(2026, 11, 1, 1, tzinfo=UTC))
    assert later.spent_this_month() == 0
    assert not later.allow(1).allowed, "the provider still reports 125 remaining"


@pytest.mark.parametrize("cost", [1, 7, 15])
def test_planned_cost_is_what_the_guard_is_asked(cost: int, tmp_path: Path) -> None:
    g = guard.CreditGuard(tmp_path / "usage.jsonl")
    assert g.allow(cost).cost == cost
