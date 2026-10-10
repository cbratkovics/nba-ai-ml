"""The pre-registered paper-bet rule, version 1 (ADR-0030).

Everything a bet needs is here as data and pure functions, so the warehouse (fct_paper_bet)
and the recommendations view apply the same rule and the ADR cites one hash. The rule is
committed before the first live date it covers and is never tuned on 2026-27; a change is a
new version with its own record.

  eligible    the prediction row qualifies: at least one prior game this season, a last-10
              minutes mean at or above the floor, inputs no older than the staleness cap
              (all three known before tip-off; evidence in
              reports/ledger_rule_evidence_2025-26.json)
  decide      over when prediction - line >= edge, under when line - prediction >= edge,
              else pass
  settle      pending until the box score is ingested; won / lost on the bet side of the
              line; push when actual equals a whole line; void when the player did not play
  profit      flat one unit at the captured American price
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

RULE: dict[str, Any] = {
    "version": "ledger_rule_v1",
    "status": "final",  # ADR-0030 approved with five amendments, 2026-10-10
    "registered_on": "2026-10-10",
    "first_live_date": "2026-10-20",
    "model": {
        "revision": "fb427de136e1d6c4b591ae30cf30488f44935182",
        "feature_version": "asof_v1",
    },
    "markets": {
        "api": ["player_points"],
        "manual": ["player_points", "player_rebounds", "player_assists"],
    },
    "books": {"api": ["draftkings"], "manual": ["draftkings", "caesars"]},
    "line_sources": {
        "api": "The Odds API free plan, one snapshot a day",
        "manual": "typed from the owner's own account",
    },
    "snapshot": {
        "scheduled_utc": "20:00",
        "scope": (
            "games whose commence_time is after the snapshot; games already started are out "
            "of scope and counted"
        ),
    },
    "eligibility": {
        "min_prior_games_season": 1,
        "min_minutes_mean_last10": 15.0,
        "max_input_staleness_days": 10,
        "why": (
            "pre-game proxy for the training population (min10): at a 15-minute floor 95.7% "
            "of the proxy rows are min10 and 86.3% of min10 rows are kept; 98.2% of min10 rows "
            "have inputs at most 10 days old (reports/ledger_rule_evidence_2025-26.json)"
        ),
    },
    "edge": {
        "threshold_by_market": {
            "player_points": 1.75,
            "player_rebounds": 0.7,
            "player_assists": 0.5,
        },
        "measured_against": "the captured line",
        "why": (
            "the edge constants already pre-registered per target on the training population "
            "(reports/policy_2025-26.json populations.min10.targets.<target>.threshold)"
        ),
    },
    "stake": {
        "units": 1.0,
        "risked": "one unit risked per bet, every bet",
        "profit_on_win": "from the captured American price on the bet side",
    },
    "settlement": {
        "pending": "the game's box score is absent from the stored logs",
        "won": "actual on the bet side of the line",
        "lost": "actual on the other side of the line",
        "push": "actual equals the line (whole-number lines only)",
        "void": (
            "the game's box score is present and the player has no row in it (did not "
            "play); a row with any minutes played, even under one, means the action stands"
        ),
    },
    "side_bias": (
        "the model is a LightGBM regressor trained on squared error (a conditional mean); "
        "a book's line sits near the median and points are right-skewed, so the rule may "
        "lean to overs; not adjusted for, reported by side in mart_ledger_by_side"
    ),
    "reading": {
        "too_early_below_settled_bets": 100,
        "then": (
            "show the 95% interval and the break-even rate implied by the actual prices; a "
            "verdict only when the interval excludes break-even"
        ),
        "power_note": (
            "separating a 55% hit rate from the 52.4% break-even of -110 takes about 1,400 "
            "settled bets: n = (1.96 * sqrt(0.55 * 0.45) / (0.55 - 0.524))^2 = 1,406"
        ),
    },
    "headline": "source api only; manual lines never enter the headline numbers",
    "reporting": (
        "in-sample (2025-26 replay) and out-of-sample (2026-27 ledger) are labelled and never "
        "pooled; no profitability claim without n and a confidence interval"
    ),
}

SIDES = ("over", "under", "pass")
OUTCOMES = ("pending", "won", "lost", "push", "void")


def rule_hash(rule: dict[str, Any] = RULE) -> str:
    """sha256 of the canonical JSON; written on every bet row and cited by ADR-0030."""
    canonical = json.dumps(rule, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def eligible(
    games_played_season: float | None,
    minutes_mean_last10: float | None,
    input_staleness_days: float | None,
    rule: dict[str, Any] = RULE,
) -> tuple[bool, list[str]]:
    e = rule["eligibility"]
    reasons = []
    if games_played_season is None or games_played_season < e["min_prior_games_season"]:
        reasons.append("no prior game this season")
    if minutes_mean_last10 is None or minutes_mean_last10 < e["min_minutes_mean_last10"]:
        reasons.append(f"last-10 minutes mean below {e['min_minutes_mean_last10']}")
    if input_staleness_days is None or input_staleness_days > e["max_input_staleness_days"]:
        reasons.append(f"inputs older than {e['max_input_staleness_days']} days")
    return (not reasons, reasons)


def decide(
    prediction: float | None,
    line: float | None,
    market: str = "player_points",
    rule: dict[str, Any] = RULE,
) -> str:
    if prediction is None or line is None or market not in rule["edge"]["threshold_by_market"]:
        return "pass"
    edge = rule["edge"]["threshold_by_market"][market]
    if prediction - line >= edge:
        return "over"
    if line - prediction >= edge:
        return "under"
    return "pass"


def settle(
    side: str,
    line: float,
    actual: float | None,
    *,
    game_ingested: bool,
    played: bool,
) -> str:
    """Outcome of a bet on `side` at `line` once the box score is (or is not) there."""
    if side not in ("over", "under"):
        raise ValueError(f"not a bet: {side!r}")
    if not game_ingested:
        return "pending"
    if not played or actual is None:
        return "void"
    if actual == line:
        return "push"
    return "won" if (actual > line) == (side == "over") else "lost"


def implied_probability(american: float) -> float:
    """Break-even win probability of an American price (the vig is in the pair, not here)."""
    return 100 / (american + 100) if american > 0 else -american / (-american + 100)


def profit_units(outcome: str, american: float, units: float = RULE["stake"]["units"]) -> float:
    if outcome == "won":
        return units * (american / 100 if american > 0 else 100 / -american)
    if outcome == "lost":
        return -units
    return 0.0  # push, void, pending


MIRROR_PATH = "frontend/lib/ledger_rule.json"


def mirror() -> dict[str, Any]:
    """The rule and its hash as the site imports them (frontend/lib/ledger_rule.json), so the
    ledger page states the registered rule without reading anything private; a test checks
    the committed file against this."""
    return {"rule_hash": rule_hash(), "rule": RULE}


def main(argv: list[str] | None = None) -> int:
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="print the rule hash, or write the site mirror")
    parser.add_argument("--mirror", action="store_true", help=f"write {MIRROR_PATH}")
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    if args.mirror:
        path = args.root / MIRROR_PATH
        path.write_text(json.dumps(mirror(), indent=2) + "\n")
        print(f"RULE {RULE['version']} {rule_hash()[:12]} -> {path}")
    else:
        print(rule_hash())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
