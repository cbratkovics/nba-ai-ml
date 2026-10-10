"""dbt warnings become labelled issues through the same protocol as the drift notifier."""

import json

from nba.warehouse import notify


def _results(statuses: dict[str, tuple[str, int]]) -> dict:
    return {
        "metadata": {"generated_at": "2026-11-01T10:30:00Z"},
        "args": {"target": "motherduck"},
        "results": [
            {
                "unique_id": f"test.nba_dbt.{name}.abc123",
                "status": status,
                "failures": failures,
                "message": f"Got {failures} result(s), configured to warn if != 0",
            }
            for name, (status, failures) in statuses.items()
        ],
    }


def test_only_warn_results_become_issues_with_stable_titles(tmp_path, capsys) -> None:
    rr = _results(
        {
            "unique_combination_slv_game_logs": ("pass", 0),
            "assert_lookback_covers_restatement_lag": ("warn", 3),
            "relationships_slv_game_logs_season": ("warn", 1),
            "assert_season_game_counts": ("fail", 1),
        }
    )
    found = notify.issues(rr)
    assert [i["title"] for i in found] == [
        "Warehouse WARN: assert_lookback_covers_restatement_lag",
        "Warehouse WARN: relationships_slv_game_logs_season",
    ]
    assert all(i["label"] == notify.WARN_LABEL for i in found)
    assert "3 row(s)" in found[0]["body"] and "motherduck" in found[0]["body"]
    assert notify.RUNBOOK in found[0]["body"] and "not blocked" in found[0]["body"]
    p = tmp_path / "run_results.json"
    p.write_text(json.dumps(rr))
    body = tmp_path / "body.md"
    assert notify.main(["--run-results", str(p), "--body-out", str(body)]) == 0
    out = capsys.readouterr().out.splitlines()
    assert (
        out[0]
        == f"NOTIFY {notify.WARN_LABEL} Warehouse WARN: assert_lookback_covers_restatement_lag"
    )
    assert len(out) == 2 and body.read_text().count("---") == 1


def test_clean_build_and_missing_file_print_nothing(tmp_path, capsys) -> None:
    p = tmp_path / "run_results.json"
    p.write_text(json.dumps(_results({"some_test": ("pass", 0)})))
    assert notify.main(["--run-results", str(p), "--body-out", str(tmp_path / "b.md")]) == 0
    assert notify.main(["--run-results", str(tmp_path / "missing.json")]) == 0
    assert capsys.readouterr().out == ""
    assert not (tmp_path / "b.md").exists()
