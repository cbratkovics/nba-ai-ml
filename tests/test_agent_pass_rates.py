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
    # Date 1: pass, ungrounded (finding dropped, so golden fails too). Date 2: never named.
    scripts = [
        _final([good]),
        _final([ungrounded]),
        _final([dict(good, text="Nobody in particular: 12.5.")]),
        _final([dict(good, text="Nobody in particular: 12.5.")]),
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
