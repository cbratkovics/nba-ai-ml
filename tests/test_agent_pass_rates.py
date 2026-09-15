import json
from datetime import date
from pathlib import Path

import pytest

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
        "dates_complete": 2,
        "dates_total": 2,
        "runs_expected": 4,
        "runs_completed": 4,
    }
    assert report["golden_dates"] == ["2026-01-13", "2026-01-14"]
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


def _date_block(runs: int, limited: int = 0, grounded: int | None = None) -> dict:
    detail = []
    for i in range(runs):
        is_limited = i < limited
        ok = (not is_limited) and (grounded is None or i - limited < grounded)
        detail.append(
            {
                "status": "rate_limited" if is_limited else ("ok" if ok else "ungrounded"),
                "grounding_pass": ok,
                "golden_pass": not is_limited,
                "n_findings": 0 if is_limited else 5,
                "n_dropped": 0,
                "model_id": "pinned",
                "tool_calls_made": 5,
                "latency_ms": 100,
                "error": None,
                "run": i + 1,
            }
        )
    return {"player_name": "P", "player_id": 1, **pass_rates._rates(detail), "runs_detail": detail}


def test_merge_replaces_dates_and_recomputes_completeness() -> None:
    golden = {
        d: {"date": d, "player_id": 1, "player_name": "P"} for d in ("2026-01-01", "2026-02-02")
    }
    existing = pass_rates.finalize(
        {
            "generated_at": "t0",
            "mode": "live",
            "model_id": "pinned",
            "fallback_model_id": None,
            "runs_per_date": 2,
            "golden_dates": sorted(golden),
            "dates": {"2026-01-01": _date_block(2), "2026-02-02": _date_block(2, limited=2)},
        }
    )
    assert existing["overall"]["complete"] is False
    assert pass_rates.incomplete_dates(existing, golden) == ["2026-02-02"]
    assert existing["overall"]["runs_completed"] == 2 and existing["overall"]["runs_expected"] == 4

    new = pass_rates.finalize(
        {
            "generated_at": "t1",
            "mode": "live",
            "model_id": "pinned",
            "fallback_model_id": None,
            "runs_per_date": 2,
            "golden_dates": sorted(golden),
            "dates": {"2026-02-02": _date_block(2, grounded=1)},
        }
    )
    merged = pass_rates.merge_reports(existing, new)
    assert merged["generated_at"] == "t1"
    assert merged["dates"]["2026-01-01"] == existing["dates"]["2026-01-01"]
    assert merged["dates"]["2026-02-02"]["statuses"] == {"ok": 1, "ungrounded": 1}
    o = merged["overall"]
    assert (o["runs"], o["grounding_pass"], o["golden_pass"], o["rate_limited"]) == (4, 3, 4, 0)
    assert o["complete"] is True and o["dates_complete"] == 2 and o["dates_total"] == 2
    assert pass_rates.incomplete_dates(merged, golden) == []
    with pytest.raises(ValueError, match="runs per date"):
        pass_rates.merge_reports(existing, dict(new, runs_per_date=3))


def test_readme_row_and_update(tmp_path: Path) -> None:
    golden = {
        d: {"date": d, "player_id": 1, "player_name": "P"} for d in ("2026-01-01", "2026-02-02")
    }
    partial = pass_rates.finalize(
        {
            "generated_at": "t0",
            "mode": "live",
            "model_id": "pinned",
            "fallback_model_id": None,
            "runs_per_date": 2,
            "golden_dates": sorted(golden),
            "dates": {
                "2026-01-01": _date_block(2, grounded=1),
                "2026-02-02": _date_block(2, limited=2),
            },
        }
    )
    assert pass_rates.readme_row(partial) == (
        "incomplete: 2 of 4 briefs completed (1 of 2 dates; 2 rate-limited); "
        "of those, grounding 1 of 2, golden 2 of 2"
    )
    full = pass_rates.merge_reports(
        partial,
        pass_rates.finalize(dict(partial, dates={"2026-02-02": _date_block(2)}, generated_at="t1")),
    )
    assert (
        pass_rates.readme_row(full)
        == "grounding 3 of 4, golden 4 of 4 (2 dates × 2 runs, `pinned`)"
    )
    readme = tmp_path / "README.md"
    readme.write_text("| row | <!-- pass-rates -->old<!-- /pass-rates --> tail |\n")
    assert pass_rates.update_readme(readme, full) is True
    assert (
        readme.read_text()
        == f"| row | <!-- pass-rates -->{pass_rates.readme_row(full)}<!-- /pass-rates --> tail |\n"
    )
    assert pass_rates.update_readme(readme, full) is False
    readme.write_text("no markers\n")
    with pytest.raises(ValueError, match="markers"):
        pass_rates.update_readme(readme, full)


def test_committed_report_and_readme_row_agree() -> None:
    report = json.loads((config.REPO_ROOT / pass_rates.PASS_RATES_PATH).read_text())
    readme = (config.REPO_ROOT / "README.md").read_text()
    expected = f"{pass_rates.README_START}{pass_rates.readme_row(report)}{pass_rates.README_END}"
    assert expected in readme, (
        "run: python -m nba.agent.pass_rates --readme README.md (or update the row)"
    )
    golden = json.loads((config.REPO_ROOT / "reports" / "agent_golden.json").read_text())
    assert report["golden_dates"] == sorted(g["date"] for g in golden["dates"])
    assert "org_REDACTED" in json.dumps(report) or "org_" not in json.dumps(report)
    assert not pass_rates.ORG_PATTERN.search(json.dumps(report).replace("org_REDACTED", ""))


def test_scrub_error_redacts_the_organisation_id() -> None:
    err = "RateLimitError: ... in organization `org_01abcXYZ` service tier ..."
    assert (
        pass_rates.scrub_error(err)
        == "RateLimitError: ... in organization `org_REDACTED` service tier ..."
    )
    assert pass_rates.scrub_error(None) is None


def test_measure_can_run_a_subset_of_dates(tmp_path: Path, game_logs) -> None:
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
    scripts = [_final([good])]
    report = pass_rates.measure(
        golden,
        1,
        lambda d: ctx,
        lambda: ScriptedChat([scripts.pop(0)]),
        pause=0,
        log=lambda s: None,
        dates=["2026-01-14"],
        runner={"runner": "local"},
    )
    assert list(report["dates"]) == ["2026-01-14"]
    assert report["golden_dates"] == ["2026-01-13", "2026-01-14"]
    assert report["dates"]["2026-01-14"]["measured"]["runner"] == "local"
    assert report["overall"]["complete"] is False and report["overall"]["dates_complete"] == 1
