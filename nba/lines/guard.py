"""The fail-closed credit guard for The Odds API free plan (ADR-0031).

Budget, from nba.config: the plan's monthly quota (500), a safety margin never planned into
(15 %, so 425 is the most that may ever be spent) and a reserve inside that (50) kept for a
manual re-capture, so the scheduled capture may spend up to 375. Before every priced call the
caller computes the cost under the documented rule and asks `allow(cost)`; the guard refuses
when the call would cross the applicable ceiling by either of two independent readings:

  1. the private usage log (every call ever made, with the cost the provider reported in
     `x-requests-last`), summed over the current calendar month (UTC); and
  2. the provider's own `x-requests-remaining` from the most recent logged answer, which
     resets on the provider's cycle, not the calendar month: if that number says fewer
     credits are left than the ceiling implies, the smaller figure wins.

Refusals are logged too, with cost 0. Nothing here retries, sleeps or loops; the capture
makes at most one attempt per event per day.

The usage log is `usage/odds_api.jsonl` in the private lines repo: one JSON object per line.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from nba import config
from nba.lines.odds_api import Usage


def budget_ceiling(use_reserve: bool = False) -> int:
    """Credits that may be spent in a month: cap × (1 − margin), minus the reserve unless the
    caller is a manual run allowed to dip into it."""
    usable = int(round(config.ODDS_MONTHLY_CREDITS * (1 - config.ODDS_SAFETY_MARGIN)))
    return usable if use_reserve else usable - config.ODDS_RESERVE_CREDITS


@dataclass
class Entry:
    at: str
    endpoint: str
    event_id: str | None
    planned_cost: int
    charged: int | None  # x-requests-last, None when the call was refused or failed
    remaining: int | None
    used: int | None
    outcome: str  # ok | refused | error
    note: str = ""
    month: str = ""

    def __post_init__(self) -> None:
        if not self.month:
            self.month = self.at[:7]


@dataclass(frozen=True)
class Verdict:
    allowed: bool
    cost: int
    ceiling: int
    spent_this_month: int
    provider_remaining: int | None
    reason: str


class CreditGuard:
    def __init__(self, log_path: Path, now: datetime | None = None):
        self.log_path = log_path
        self.now = now or datetime.now(UTC)
        self.entries: list[Entry] = []
        if log_path.exists():
            for line in log_path.read_text().splitlines():
                if line.strip():
                    self.entries.append(Entry(**json.loads(line)))

    # -- readings ---------------------------------------------------------------------------
    @property
    def month(self) -> str:
        return self.now.strftime("%Y-%m")

    def spent_this_month(self) -> int:
        return sum((e.charged or 0) for e in self.entries if e.month == self.month)

    def provider_remaining(self) -> int | None:
        for e in reversed(self.entries):
            if e.remaining is not None:
                return e.remaining
        return None

    # -- the decision -------------------------------------------------------------------------
    def allow(self, cost: int, use_reserve: bool = False) -> Verdict:
        ceiling = budget_ceiling(use_reserve)
        spent = self.spent_this_month()
        remaining = self.provider_remaining()
        if cost <= 0:
            return Verdict(True, cost, ceiling, spent, remaining, "free call")
        if spent + cost > ceiling:
            return Verdict(
                False,
                cost,
                ceiling,
                spent,
                remaining,
                f"would spend {spent + cost} of a {ceiling} ceiling this month",
            )
        # The provider's counter covers the margin too: never let it fall below the credits
        # the margin keeps out of reach (cap − usable), plus the reserve on a scheduled run.
        floor = config.ODDS_MONTHLY_CREDITS - budget_ceiling(use_reserve)
        if remaining is not None and remaining - cost < floor:
            return Verdict(
                False,
                cost,
                ceiling,
                spent,
                remaining,
                f"provider reports {remaining} remaining; {cost} would leave less than "
                f"the {floor} kept back",
            )
        return Verdict(True, cost, ceiling, spent, remaining, "within budget")

    # -- the log ------------------------------------------------------------------------------
    def record(
        self,
        endpoint: str,
        event_id: str | None,
        planned_cost: int,
        outcome: str,
        usage: Usage | None = None,
        note: str = "",
    ) -> Entry:
        entry = Entry(
            at=(usage.at if usage else datetime.now(UTC).isoformat(timespec="seconds")),
            endpoint=endpoint,
            event_id=event_id,
            planned_cost=planned_cost,
            charged=(usage.last if usage and outcome == "ok" else None),
            remaining=(usage.remaining if usage else None),
            used=(usage.used if usage else None),
            outcome=outcome,
            note=note,
        )
        self.entries.append(entry)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a") as fh:
            fh.write(json.dumps(asdict(entry)) + "\n")
        return entry

    def summary(self) -> dict[str, Any]:
        return {
            "month": self.month,
            "spent_this_month": self.spent_this_month(),
            "ceiling_scheduled": budget_ceiling(False),
            "ceiling_manual": budget_ceiling(True),
            "monthly_credits": config.ODDS_MONTHLY_CREDITS,
            "provider_remaining": self.provider_remaining(),
            "calls_this_month": sum(1 for e in self.entries if e.month == self.month),
            "refusals_this_month": sum(
                1 for e in self.entries if e.month == self.month and e.outcome == "refused"
            ),
        }
