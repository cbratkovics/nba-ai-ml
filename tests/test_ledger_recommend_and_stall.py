"""The recommendations view applies the rule and flags availability; manual rows become lines in
the captured layout; the source-stall streak and its issue are distinct from no-schedule."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd

from nba import config, nightly
from nba.drift import notify
from nba.ledger import recommend, rule
from nba.lines import manual


def _lines(
    player_id,
    name,
    line=24.5,
    snapshot="2026-10-20T20:00:00+00:00",
    tip="2026-10-21T00:00:00+00:00",
):
    return pd.DataFrame(
        [
            {
                "capture_id": "c",
                "snapshot_at_utc": pd.Timestamp(snapshot),
                "source": "api",
                "book": "draftkings",
                "event_id": "e",
                "commence_time_utc": pd.Timestamp(tip),
                "game_date": "2026-10-20",
                "home_team": "LAL",
                "away_team": "BOS",
                "team": "LAL",
                "player_name_raw": name,
                "player_id": player_id,
                "match_method": "team",
                "market": "player_points",
                "line": line,
                "over_price": -115,
                "under_price": -105,
                "line_last_update_utc": pd.Timestamp(snapshot),
            }
        ]
    )


def _preds(pid, pred=27.0, gps=3, mins=30.0, stale=2.0, generated="2026-10-20T10:00:00+00:00"):
    return pd.DataFrame(
        [
            {
                "player_id": pid,
                "pred_pts": pred,
                "pred_reb": 5.0,
                "pred_ast": 4.0,
                "games_played_season": gps,
                "minutes_mean_last10": mins,
                "input_staleness_days": stale,
                "generated_at": generated,
                "tip_off_utc": pd.Timestamp("2026-10-21T00:00:00+00:00"),
            }
        ]
    )


def test_recommendation_applies_the_rule_and_flags_availability() -> None:
    rows = recommend.build(_lines(7, "A Player"), _preds(7), date(2026, 10, 20))
    assert len(rows) == 1 and rows[0]["decision"] == "over" and rows[0]["price"] == -115
    assert rows[0]["edge"] == 2.5 and rows[0]["availability"] == "unknown" and rows[0]["eligible"]
    # Ineligible: a season debut passes nothing even with a large edge.
    rows = recommend.build(_lines(7, "A Player"), _preds(7, gps=0), date(2026, 10, 20))
    assert rows[0]["decision"] == "pass" and "no prior game" in rows[0]["reasons"][0]
    # Ordering rule: a prediction stamped after the line disqualifies the row.
    rows = recommend.build(
        _lines(7, "A Player"), _preds(7, generated="2026-10-20T21:00:00+00:00"), date(2026, 10, 20)
    )
    assert rows[0]["decision"] == "pass" and "after the line" in rows[0]["reasons"][-1]
    # A line read after tip-off disqualifies too; an unmatched line is not shown; no
    # prediction passes.
    rows = recommend.build(
        _lines(7, "A Player", snapshot="2026-10-21T01:00:00+00:00"), _preds(7), date(2026, 10, 20)
    )
    assert rows[0]["decision"] == "pass" and "after tip-off" in rows[0]["reasons"][-1]
    assert recommend.build(_lines(None, "Nobody"), _preds(7), date(2026, 10, 20)) == []
    rows = recommend.build(_lines(8, "Other"), _preds(7), date(2026, 10, 20))
    assert rows[0]["decision"] == "pass" and rows[0]["prediction"] is None


def test_recommendation_file_carries_rule_hash_and_coverage(tmp_path: Path) -> None:
    root = tmp_path
    (root / config.PREDICTIONS_DIR).mkdir()
    (root / config.PREDICTIONS_DIR / "2026-10-20.parquet").write_bytes(b"")
    (root / config.CAPTURES_DIR).mkdir()
    (root / config.CAPTURES_DIR / "2026-10-20.json").write_text(json.dumps({"status": "captured"}))
    rows = recommend.build(_lines(7, "A Player"), _preds(7), date(2026, 10, 20))
    path = recommend.write(root, date(2026, 10, 20), rows, push=False)
    payload = json.loads(path.read_text())
    assert payload["rule_hash"] == rule.rule_hash() and payload["n_bets"] == 1
    assert payload["coverage"] == {
        "nights_captured": 1,
        "game_nights_with_slate": 1,
        "bets_pending": None,
    }
    assert "no availability information" in payload["availability_note"]


def test_manual_rows_become_lines_and_are_matched(tmp_path: Path, game_logs: pd.DataFrame) -> None:
    name = sorted(game_logs.loc[game_logs["team"] == "BOS", "player_name"].unique())[0]
    csv = tmp_path / "m.csv"
    csv.write_text(
        manual.template_text()
        + f"2026-10-20T20:05:00Z,2026-10-20,caesars,{name},BOS,player_points,27.5,-115,-105,\n"
        + "2026-10-20T20:06:00Z,2026-10-20,draftkings,Nobody Known,BOS,player_rebounds,6.5,"
        + "-110,-110,\n"
    )
    lines = manual.to_lines(manual.load(csv), game_logs)
    assert list(lines.columns) == list(__import__("nba.lines.capture", fromlist=["x"]).LINE_COLUMNS)
    assert lines["source"].tolist() == ["manual", "manual"]
    assert lines["match_method"].tolist() == ["team", "unmatched"]
    assert (
        lines["capture_id"].iloc[0] == "2026-10-20-manual"
        and lines["game_date"].iloc[0] == "2026-10-20"
    )
    written = manual.ingest(csv, tmp_path, game_logs, push=False)
    assert [p.name for p in written] == ["2026-10-20_manual.parquet"]
    # Re-ingesting the same rows does not duplicate them.
    manual.ingest(csv, tmp_path, game_logs, push=False)
    assert len(pd.read_parquet(written[0])) == 2


def test_stall_streak_and_its_issue() -> None:
    lines = [
        {"date": "2026-10-20", "n_predicted": 480, "n_with_actuals": 470},
        {"date": "2026-10-21", "n_predicted": 300, "n_with_actuals": 0},
        {"date": "2026-10-22", "n_predicted": 250, "n_with_actuals": 0},
    ]
    assert nightly.stall_streak(lines) == 2
    assert nightly.stall_streak(lines[:1]) == 0
    assert nightly.stall_streak([]) == 0
    assert nightly.stall_streak([{"date": "x", "n_predicted": 0, "n_with_actuals": 0}]) == 0
    found = notify.issues(
        {
            "date": "2026-10-23",
            "source": {"stalled": True, "stall_streak": 2, "last_residual_date": "2026-10-22"},
        }
    )
    assert [i["label"] for i in found] == [notify.STALL_LABEL]
    assert "2 consecutive" in found[0]["body"] and "stay pending" in found[0]["body"]
    assert notify.issues({"date": "x", "source": {"stalled": False, "stall_streak": 1}}) == []
