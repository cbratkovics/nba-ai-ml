"""The site's all-rows numbers are derived from the committed replay file and never drift."""

import hashlib
import json

from nba import config
from nba.models import evaluate

ROOT = config.REPO_ROOT


def test_frontend_summary_matches_committed_report() -> None:
    report = ROOT / evaluate.all_rows_report_path()
    expected = evaluate.all_rows_summary(report)
    committed = json.loads((ROOT / evaluate.ALL_ROWS_SUMMARY_PATH).read_text())
    assert committed == expected, "run: python -m nba.models.evaluate all-rows"
    assert committed["source_sha256"] == hashlib.sha256(report.read_bytes()).hexdigest()
    assert committed["season"] == config.HOLDOUT_SEASON
    assert committed["n"] == sum(d["n"] for d in json.loads(report.read_text())["days"])


def test_summary_is_row_weighted(tmp_path) -> None:
    daily = {
        "season": "2025-26",
        "population": "p",
        "targets": ["pts"],
        "n_dates": 2,
        "first_date": "a",
        "last_date": "b",
        "days": [
            {"date": "a", "n": 1, "model": {"pts": 1.0}, "baseline_last10": {"pts": 3.0}},
            {"date": "b", "n": 3, "model": {"pts": 5.0}, "baseline_last10": {"pts": 3.0}},
        ],
    }
    path = tmp_path / "r.json"
    path.write_text(json.dumps(daily))
    s = evaluate.all_rows_summary(path)
    assert (
        s["n"] == 4 and s["model_mae"] == {"pts": 4.0} and s["baseline_last10_mae"] == {"pts": 3.0}
    )
    assert s["baseline_wins"] == ["pts"]


def test_summary_rounds_weighted_means_at_a_stable_serialization_boundary(tmp_path) -> None:
    daily = {
        "season": "2025-26",
        "population": "p",
        "targets": ["pts"],
        "n_dates": 3,
        "days": [
            {"n": 1, "model": {"pts": 0.1}, "baseline_last10": {"pts": 0.3}},
            {"n": 1, "model": {"pts": 0.2}, "baseline_last10": {"pts": 0.2}},
            {"n": 1, "model": {"pts": 0.3}, "baseline_last10": {"pts": 0.1}},
        ],
    }
    path = tmp_path / "r.json"
    path.write_text(json.dumps(daily))

    summary = evaluate.all_rows_summary(path)

    assert summary["model_mae"] == {"pts": 0.2}
    assert summary["baseline_last10_mae"] == {"pts": 0.2}
