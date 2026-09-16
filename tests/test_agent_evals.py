import json
from datetime import date
from pathlib import Path

from nba import config
from nba.agent import evals, loop, tools
from nba.storage import local


def test_numbers_in_text_skips_dates_and_ids() -> None:
    text = (
        "On 2026-04-12 game 0022501186 Nikola Jokic scored 40 (pred 24.5), MAE 4.766; 1,234 rows."
    )
    assert evals.numbers_in_text(text) == [40.0, 24.5, 4.766, 1234.0]
    # Hyphenated labels are not measurements, whichever dash the model used.
    assert evals.numbers_in_text("30-day MAE, last\u201110 baseline, season 2024-25: 7 games") == [
        7.0
    ]
    assert evals.numbers_in_text("days_stale = -32 (32 days).") == [-32.0, 32.0]


def test_is_grounded_with_tolerance_and_nested_values() -> None:
    f = {
        "text": "Predicted 24.5, actual 40, residual 15.5.",
        "evidence": {"values": {"top": [{"predicted": 24.504, "actual": 40, "residual": "15.5"}]}},
    }
    ok, missing = evals.is_grounded(f)
    assert ok and missing == []
    f["text"] = "Predicted 24.5, actual 41."
    ok, missing = evals.is_grounded(f)
    assert not ok and missing == [41.0]
    # The cited call's arguments count as evidence: a 30-day window named in the text.
    f = {"text": "MAE over 30 days: 4.9", "evidence": {"args": {"days": 30}, "values": {"m": 4.9}}}
    assert evals.is_grounded(f) == (True, [])
    # Sign-insensitive: "32 days ahead" is grounded by days_stale = -32.
    assert evals.is_grounded({"text": "32 days ahead", "evidence": {"values": {"d": -32}}})[0]


def test_apply_grounding_marks_and_drops() -> None:
    brief = {
        "status": "ok",
        "findings": [
            {"kind": "a", "text": "x is 3", "evidence": {"values": {"x": 3}}},
            {"kind": "b", "text": "y is 4", "evidence": {"values": {"y": 5}}},
        ],
    }
    out = evals.apply_grounding(brief)
    assert out["status"] == "ungrounded" and [f["kind"] for f in out["findings"]] == ["a"]
    assert out["dropped_findings"][0]["kind"] == "b" and out["grounding"] == {
        "checked": 2,
        "dropped": 1,
    }
    untouched = evals.apply_grounding({"status": "agent_unavailable", "findings": []})
    assert untouched["status"] == "agent_unavailable"


def test_golden_hit_by_name_or_id() -> None:
    brief = {
        "findings": [
            {
                "text": "Largest miss: Nikola Jokic 40 vs 24.5",
                "evidence": {"values": {"player_id": 203999}},
            }
        ]
    }
    assert evals.golden_hit(brief, "nikola jokic")
    assert evals.golden_hit(brief, "Someone Else", player_id=203999)
    assert not evals.golden_hit(brief, "Someone Else", player_id=1)


def test_run_from_traces_and_report(tmp_path: Path, game_logs) -> None:
    root = tmp_path / "root"
    local.write_per_season(game_logs, root / config.DATA_DIR)
    ctx = tools.ToolContext(root=root, data_dir=root / config.DATA_DIR, run_date=date(2026, 1, 15))
    # Record a trace with a scripted chat, then evaluate it via replay.
    from tests.test_agent_loop import ScriptedChat, _final, _tool_call

    chat = ScriptedChat(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [_tool_call("c1", "list_data_gaps", {})],
            },
            _final(
                [
                    {
                        "kind": "residual_outlier",
                        "severity": "warning",
                        "evidence": {
                            "tool": "get_residuals",
                            "args": {"date": "2026-01-14"},
                            "values": {"player_name": "Test Player", "residual": 12.5},
                        },
                        "text": "Test Player missed by 12.5.",
                    }
                ]
            ),
        ]
    )
    brief, trace = loop.run_agent(ctx, date(2026, 1, 14), chat)
    traces = tmp_path / "traces"
    traces.mkdir()
    (traces / "2026-01-14.trace.json").write_text(json.dumps(trace.to_dict()))
    golden = {
        "2026-01-14": {
            "date": "2026-01-14",
            "player_id": 1,
            "player_name": "Test Player",
            "resid_pts": 12.5,
        }
    }
    results = evals.run_from_traces(traces, golden, lambda t: ctx)
    assert len(results) == 1 and results[0]["grounding_pass"] and results[0]["golden"]["pass"]
    payload = evals.write_report(
        results, tmp_path / "reports" / "agent_evals.json", mode="trace_replay"
    )
    assert payload["golden_pass"] == 1 and payload["golden_total"] == 1 and payload["n_briefs"] == 1
    assert (
        json.loads((tmp_path / "reports" / "agent_evals.json").read_text())["mode"]
        == "trace_replay"
    )
    # A golden player the brief does not name fails.
    miss = {"2026-01-14": {"date": "2026-01-14", "player_id": 2, "player_name": "Other Guy"}}
    assert evals.run_from_traces(traces, miss, lambda t: ctx)[0]["golden"]["pass"] is False


def test_committed_traces_replay_cleanly() -> None:
    """Every trace under tests/traces must replay into a grounded brief with the golden player."""
    traces_dir = evals.TRACES_DIR
    golden = evals.load_golden()
    paths = sorted(traces_dir.glob("*.trace.json")) if traces_dir.exists() else []
    if not paths:
        return  # no live traces recorded yet
    for path in paths:
        text = path.read_text()
        assert "gsk_" not in text, f"{path} contains a key-like string"
        assert len(text.encode()) <= loop.MAX_TRACE_BYTES, f"{path} is larger than 100 KB"
    ctx = tools.ToolContext(root=Path("."), data_dir=config.DATA_DIR, run_date=date(2026, 1, 1))
    results = evals.run_from_traces(traces_dir, golden, lambda t: ctx)
    assert all(r["status"] in ("ok", "ungrounded") for r in results)
    for r in results:
        if "golden" in r:
            assert r["golden"]["pass"], f"golden miss on {r['date']}: {r['golden']}"


def test_decision_and_drift_facts() -> None:
    brief = {
        "findings": [
            {
                "kind": "decision_policy",
                "text": "Training population pts hit rate to date 0.66 over 5,138 calls.",
                "evidence": {
                    "tool": "get_rolling_metrics",
                    "args": {"days": 30},
                    "values": {"hit_rate": 0.6625, "n_resolved": 5138},
                },
            },
            {
                "kind": "drift",
                "text": "Drift status is ok as of 2026-03-11; no feature flagged.",
                "evidence": {
                    "tool": "get_daily_report",
                    "args": {"date": "2026-03-11"},
                    "values": {"status": "ok", "n_flagged": 0},
                },
            },
        ]
    }
    decision = {"tool": "get_rolling_metrics", "value": 0.6625}
    drift = {"tool": "get_daily_report", "status": "ok"}
    assert evals.decision_hit(brief, decision) is True
    assert evals.decision_hit(brief, {"tool": "get_rolling_metrics", "value": 0.51}) is False
    assert evals.decision_hit(brief, {"tool": "get_residuals", "value": 0.6625}) is False
    assert evals.decision_hit(brief, None) is None
    # A percentage still counts.
    pct = {
        "findings": [
            {"text": "hit rate 66%", "evidence": {"tool": "get_rolling_metrics", "values": {}}}
        ]
    }
    assert evals.decision_hit(pct, decision) is True
    assert evals.drift_hit(brief, drift) is True
    assert evals.drift_hit(brief, {"tool": "get_daily_report", "status": "hold"}) is False
    assert evals.drift_hit(brief, None) is None
    golden = {
        "player_id": 1,
        "player_name": "Nobody",
        "decision_fact": decision,
        "drift_fact": drift,
    }
    result = evals.golden_result(brief, golden)
    assert result["decision_pass"] and result["drift_pass"] and not result["player_pass"]
    assert result["pass"] is False
    golden_v1 = {"player_id": 1, "player_name": "Nobody"}
    r1 = evals.golden_result({"findings": [{"text": "Nobody scored", "evidence": {}}]}, golden_v1)
    assert r1["pass"] is True and r1["decision_pass"] is None and r1["drift_pass"] is None
