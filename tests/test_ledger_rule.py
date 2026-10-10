"""The pre-registered rule (ADR-0030): eligibility, the decision against the line, settlement
including push / void / pending, the price arithmetic, and the hash the bets carry."""

from __future__ import annotations

import json

import pytest

from nba import config
from nba.ledger import rule


def test_rule_cites_the_frozen_model_and_the_evidence() -> None:
    assert rule.RULE["model"]["revision"] == config.MODEL_REVISION
    assert rule.RULE["markets"]["api"] == list(config.ODDS_MARKETS)
    assert rule.RULE["books"]["api"] == list(config.ODDS_BOOKMAKERS)
    assert rule.RULE["snapshot"]["scheduled_utc"] == config.CAPTURE_SNAPSHOT_UTC
    evidence = json.loads(
        (config.REPO_ROOT / "reports" / "ledger_rule_evidence_2025-26.json").read_text()
    )
    floor = rule.RULE["eligibility"]["min_minutes_mean_last10"]
    row = next(r for r in evidence["proxy"]["table"] if r["minutes_floor"] == floor)
    assert round(row["precision"], 3) == 0.957 and round(row["recall"], 3) == 0.863
    cap = str(rule.RULE["eligibility"]["max_input_staleness_days"])
    assert round(1 - evidence["staleness"]["share_above"][cap], 3) == 0.982
    policy = json.loads((config.REPO_ROOT / "reports" / "policy_2025-26.json").read_text())
    for market, target in (
        ("player_points", "pts"),
        ("player_rebounds", "reb"),
        ("player_assists", "ast"),
    ):
        assert (
            rule.RULE["edge"]["threshold_by_market"][market]
            == policy["populations"]["min10"]["targets"][target]["threshold"]
        )
    assert len(rule.rule_hash()) == 64 and rule.rule_hash() == rule.rule_hash()


def test_eligibility_reasons() -> None:
    assert rule.eligible(3, 28.0, 2) == (True, [])
    ok, why = rule.eligible(0, 28.0, 2)
    assert not ok and why == ["no prior game this season"]
    ok, why = rule.eligible(5, 14.9, 11)
    assert not ok and len(why) == 2
    assert not rule.eligible(None, None, None)[0]


def test_decision_against_the_line() -> None:
    assert rule.decide(27.0, 25.0) == "over"  # edge 2.0 >= 1.75
    assert rule.decide(26.7, 25.0) == "pass"  # 1.7 < 1.75
    assert rule.decide(23.25, 25.0) == "under"  # exactly 1.75
    assert rule.decide(None, 25.0) == "pass" and rule.decide(25.0, None) == "pass"
    assert rule.decide(7.3, 6.5, "player_rebounds") == "over"  # 0.8 >= 0.7
    assert rule.decide(7.1, 6.5, "player_rebounds") == "pass"
    assert rule.decide(3.0, 2.5, "player_threes") == "pass"  # not a rule market


@pytest.mark.parametrize(
    "side,line,actual,ingested,played,expected",
    [
        ("over", 25.5, None, False, False, "pending"),
        ("over", 25.5, 26, True, True, "won"),
        ("over", 25.5, 25, True, True, "lost"),
        ("under", 25.5, 25, True, True, "won"),
        ("under", 25.0, 25, True, True, "push"),
        ("over", 25.5, None, True, False, "void"),
    ],
)
def test_settlement(side, line, actual, ingested, played, expected) -> None:
    assert rule.settle(side, line, actual, game_ingested=ingested, played=played) == expected


def test_settlement_refuses_a_pass() -> None:
    with pytest.raises(ValueError):
        rule.settle("pass", 25.5, 26, game_ingested=True, played=True)


def test_prices() -> None:
    assert rule.implied_probability(-110) == pytest.approx(0.5238, abs=1e-4)
    assert rule.implied_probability(+150) == pytest.approx(0.4)
    assert rule.profit_units("won", -110) == pytest.approx(0.9091, abs=1e-4)
    assert rule.profit_units("won", +120) == pytest.approx(1.2)
    assert rule.profit_units("lost", -110) == -1.0
    assert rule.profit_units("push", -110) == 0.0 == rule.profit_units("void", +100)


def test_site_mirror_matches_the_rule_and_its_hash() -> None:
    """frontend/lib/ledger_rule.json is the rule the ledger page states (ADR-0033)."""
    import json

    from nba import config

    committed = json.loads((config.REPO_ROOT / rule.MIRROR_PATH).read_text())
    assert committed == rule.mirror(), "run: python -m nba.ledger.rule --mirror"
    assert committed["rule_hash"] == rule.rule_hash()
    data_ts = (config.REPO_ROOT / "frontend" / "lib" / "data.ts").read_text()
    assert "export const LEDGER_SEASON = '2026-27'" in data_ts
    assert rule.RULE["first_live_date"] == config.REGULAR_SEASON_WINDOWS["2026-27"][0]
