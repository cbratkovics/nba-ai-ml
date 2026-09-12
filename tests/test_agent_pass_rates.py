import json
from datetime import date
from pathlib import Path

from nba import config
from nba.agent import pass_rates, tools
from nba.storage import local
from tests.test_agent_loop import ScriptedChat, _final


def test_measure_counts_grounding_and_golden_per_date(tmp_path: Path, game_logs) -> None:
    root = tmp_path / "root"
    local.write_per_season(game_logs, root / config.DATA_DIR)
    ctx = tools.ToolContext(root=root, data_dir=root / config.DATA_DIR, run_date=date(2026, 1, 15))
    golden = {
        "2026-01-14": {"date": "2026-01-14", "player_id": 1, "player_name": "Test Player"},
        "2026-01-13": {"date": "2026-01-13", "player_id": 2, "player_name": "Other Guy"},
    }
    good = {
        "kind": "residual_outlier",
        "severity": "warning",
        "evidence": {"tool": "get_residuals", "args": {}, "values": {"residual": 12.5}},
        "text": "Test Player missed by 12.5.",
    }
    ungrounded = dict(good, text="Test Player missed by 99.")
    # Dates run in sorted order. 2026-01-13: grounded but never names its player.
    # 2026-01-14: one pass, one ungrounded (finding dropped, so golden fails too).
    scripts = [
        _final([dict(good, text="Nobody in particular: 12.5.")]),
        _final([dict(good, text="Nobody in particular: 12.5.")]),
        _final([good]),
        _final([ungrounded]),
    ]

    def chat_factory():
        return ScriptedChat([scripts.pop(0)])

    logged = []
    report = pass_rates.measure(golden, 2, lambda d: ctx, chat_factory, pause=0, log=logged.append)
    d1, d2 = report["dates"]["2026-01-14"], report["dates"]["2026-01-13"]
    assert (d1["grounding_pass"], d1["golden_pass"], d1["runs"]) == (1, 1, 2)
    assert d1["statuses"] == {"ok": 1, "ungrounded": 1}
    assert (d2["grounding_pass"], d2["golden_pass"]) == (2, 0)
    assert report["overall"] == {
        "runs": 4,
        "grounding_pass": 3,
        "grounding_rate": 0.75,
        "golden_pass": 1,
        "golden_rate": 0.25,
        "rate_limited": 0,
        "complete": True,
        "statuses": {"ok": 3, "ungrounded": 1},
        "models": {"scripted-model": 4},
    }
    assert len(logged) == 4 and logged[0].startswith("PASSRATE 2026-01-13 run 1/2")
    assert json.dumps(report)  # serializable


def test_unavailable_brief_fails_both(tmp_path: Path, game_logs) -> None:
    root = tmp_path / "root"
    local.write_per_season(game_logs, root / config.DATA_DIR)
    ctx = tools.ToolContext(root=root, data_dir=root / config.DATA_DIR, run_date=date(2026, 1, 15))

    class Broken:
        model_id = "broken"

        def complete(self, messages, tool_schemas):
            raise ConnectionError("down")

    r = pass_rates.run_once(ctx, date(2026, 1, 14), {"player_name": "X", "player_id": 1}, Broken())
    assert r["status"] == "agent_unavailable" and not r["grounding_pass"] and not r["golden_pass"]
    assert "down" in r["error"]


def _rate_limit_error():
    import httpx
    from groq import RateLimitError

    resp = httpx.Response(429, request=httpx.Request("POST", "http://groq.test"))
    return RateLimitError("tokens per day", response=resp, body=None)


def test_rate_limited_run_is_recorded_separately(tmp_path: Path, game_logs) -> None:
    root = tmp_path / "root"
    local.write_per_season(game_logs, root / config.DATA_DIR)
    ctx = tools.ToolContext(root=root, data_dir=root / config.DATA_DIR, run_date=date(2026, 1, 15))

    class Capped:
        model_id = "pinned"

        def complete(self, messages, tool_schemas):
            raise _rate_limit_error()

    golden = {"2026-01-14": {"date": "2026-01-14", "player_id": 1, "player_name": "Test Player"}}
    report = pass_rates.measure(golden, 2, lambda d: ctx, Capped, pause=0, log=lambda s: None)
    d = report["dates"]["2026-01-14"]
    assert d["statuses"] == {"rate_limited": 2}
    assert (d["grounding_pass"], d["golden_pass"], d["rate_limited"]) == (0, 0, 2)
    assert report["overall"]["rate_limited"] == 2 and report["overall"]["complete"] is False
    assert all(r["status"] == "rate_limited" for r in d["runs_detail"])
    # Other provider failures stay agent_unavailable.
    assert not pass_rates._is_rate_limit("ConnectionError: down")


def test_no_fallback_flag_builds_a_single_model_chat(
    tmp_path: Path, game_logs, monkeypatch
) -> None:
    root = tmp_path / "root"
    local.write_per_season(game_logs, root / config.DATA_DIR)
    golden_path = tmp_path / "golden.json"
    golden_path.write_text(
        json.dumps({"dates": [{"date": "2026-01-14", "player_id": 1, "player_name": "P"}]})
    )
    built = []

    class FakeGroqChat:
        model_id = "pinned"

        def __init__(self, fallback_model_id="backup"):
            built.append(fallback_model_id)

        def complete(self, messages, tool_schemas):
            raise _rate_limit_error()

    monkeypatch.setattr(pass_rates.loop, "GroqChat", FakeGroqChat)
    out = tmp_path / "pass_rates.json"
    rc = pass_rates.main(
        [
            "--runs",
            "1",
            "--pause",
            "0",
            "--no-fallback",
            "--golden",
            str(golden_path),
            "--out",
            str(out),
            "--root",
            str(root),
        ]
    )
    assert rc == 0 and built == [None]
    report = json.loads(out.read_text())
    assert report["fallback_enabled"] is False and report["fallback_model_id"] is None
    assert report["overall"]["rate_limited"] == 1 and report["overall"]["complete"] is False
    # Default keeps the fallback.
    built.clear()
    pass_rates.main(
        [
            "--runs",
            "1",
            "--pause",
            "0",
            "--golden",
            str(golden_path),
            "--out",
            str(out),
            "--root",
            str(root),
        ]
    )
    assert built == ["backup"]
