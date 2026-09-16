"""The decision policy: call and outcome rules, threshold selection, bands, the committed
artifact's internal consistency, the site summary derived from it, and the slate decisions."""

from __future__ import annotations

import hashlib
import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

from nba import config
from nba.decisions import decide, evaluate, policy

ROOT = config.REPO_ROOT
REPORT = ROOT / evaluate.report_path()


def test_threshold_grid_is_rounded_and_inclusive() -> None:
    grid = policy.threshold_grid("pts")
    assert grid[0] == 0.0 and grid[-1] == config.POLICY_THRESHOLD_GRID["pts"][1]
    assert all(round(t, 2) == t for t in grid)
    assert len(policy.threshold_grid("reb")) == 41


def test_calls_and_outcomes() -> None:
    edge = pd.Series([2.0, -2.0, 0.5, -0.5, 0.0, np.nan])
    none = "no_call"
    assert list(policy.calls(edge, 1.0)) == ["over", "under", none, none, none, none]
    assert list(policy.calls(edge, 0.0)) == ["over", "under", "over", "under", none, none]
    call = pd.Series(["over", "over", "under", "under", "over", "no_call"])
    actual = pd.Series([12.0, 8.0, 8.0, 12.0, 10.0, 30.0])
    baseline = pd.Series([10.0] * 6)
    assert list(policy.outcomes(call, actual, baseline)) == [
        "hit",
        "miss",
        "hit",
        "miss",
        "push",
        None,
    ]
    # No actual yet: no outcome, whatever the call.
    assert list(policy.outcomes(call, pd.Series([np.nan] * 6), baseline)) == [None] * 6


def _frame(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = rng.uniform(5, 25, n)
    edge = rng.normal(0, 3, n)
    actual = (
        base
        + np.sign(edge) * rng.uniform(0, 4, n) * (rng.uniform(size=n) < 0.7)
        + rng.normal(0, 2, n)
    )
    df = pd.DataFrame(
        {
            "player_id": np.arange(n),
            "game_id": [f"00225{i:05d}" for i in range(n)],
            "has_actual": True,
            "baseline_defined": True,
            "in_metrics_population": rng.uniform(size=n) < 0.8,
            "model_revision": "m",
            "feature_version": "asof_v1",
            "dataset_revision": "d",
        }
    )
    for t in config.TARGETS:
        df[f"pred_{t}"] = base + edge
        df[f"baseline_last10_{t}"] = base
        df[f"baseline_season_{t}"] = base + rng.normal(0, 1, n)
        df[f"actual_{t}"] = np.round(actual)
    df.loc[0, "baseline_season_pts"] = np.nan  # a season debut: the season-mean sign abstains
    return df


def test_curve_arithmetic_and_selection() -> None:
    df = _frame()
    curve = policy.coverage_curve(df, "pts")
    assert [p["threshold"] for p in curve] == policy.threshold_grid("pts")
    for p in curve:
        assert p["n_called"] == p["n_resolved"] + p["n_push"]
        assert p["coverage"] == p["n_called"] / len(df)
        assert p["hit_rate"] is None or 0 <= p["hit_rate"] <= 1
        assert p["season_mean_same_rows_n"] <= p["n_resolved"]
    coverages = [p["coverage"] for p in curve]
    assert coverages == sorted(coverages, reverse=True)
    chosen = policy.select_threshold(curve, 0.25)
    assert next(p for p in curve if p["threshold"] == chosen)["coverage"] >= 0.25
    later = [p for p in curve if p["threshold"] > chosen]
    assert all(p["coverage"] < 0.25 for p in later)
    # A floor nothing on the grid reaches falls back to the smallest threshold.
    assert policy.select_threshold(curve, 1.5) == 0.0


def test_bands_are_residual_quantiles_with_nominal_coverage() -> None:
    residual = pd.Series(np.arange(1, 101, dtype="float64"))
    q = policy.bands(residual)
    assert q["q25"] < q["q75"] and q["q10"] < q["q25"] and q["q75"] < q["q90"]
    cov = policy.band_coverage(residual, q)
    assert cov["coverage_50"] == pytest.approx(0.5, abs=0.02)
    assert cov["coverage_80"] == pytest.approx(0.8, abs=0.02)


def test_artifact_builds_and_validates_on_a_synthetic_frame() -> None:
    df = _frame()
    artifact = policy.build_artifact(
        df, "2025-26", generated_at="t", git_sha="s", input_description={"source": "test"}
    )
    assert policy.validate_artifact(artifact) == []
    assert artifact["in_sample"] is True
    pops = artifact["populations"]
    assert pops["all"]["n"] == len(df)
    assert pops["min10"]["n"] == int(df["in_metrics_population"].sum())
    for block in pops.values():
        for t in config.TARGETS:
            tb = block["targets"][t]
            assert tb["baselines"]["coin_flip"]["hit_rate"] == 0.5
            assert tb["verdict"]
            assert len(tb["coverage_curve"]) == len(policy.threshold_grid(t))
    broken = json.loads(json.dumps(artifact))
    broken["populations"]["all"]["targets"]["pts"]["threshold"] = 99.0
    assert any("not on the grid" in p for p in policy.validate_artifact(broken))


def test_verdict_says_when_the_model_loses() -> None:
    chosen = {
        "hit_rate": 0.52,
        "n_resolved": 1000,
        "season_mean_same_rows_hit_rate": 0.56,
    }
    beats, text = policy.verdict("pts", "all", chosen)
    assert not beats and "not recommended" in text and "season-mean sign hits 56.0%" in text
    winning = {"hit_rate": 0.62, "n_resolved": 5000, "season_mean_same_rows_hit_rate": 0.56}
    beats, text = policy.verdict("pts", "min10", winning)
    assert beats and "above the season-mean sign" in text


@pytest.mark.skipif(not REPORT.exists(), reason="policy artifact not written yet")
def test_committed_artifact_is_consistent() -> None:
    artifact = json.loads(REPORT.read_text())
    assert policy.validate_artifact(artifact) == []
    assert artifact["season"] == config.HOLDOUT_SEASON
    assert artifact["model_revision"] == config.MODEL_REVISION
    assert artifact["model_commit"] == config.MODEL_COMMIT
    assert artifact["min_coverage"] == config.POLICY_MIN_COVERAGE
    assert artifact["threshold_grid"] == {t: policy.threshold_grid(t) for t in config.TARGETS}
    # The two populations the brief fixes: the replay's training-population rows and all rows
    # with a box score and a baseline (reports/replay_<season>.json n_restricted / n_with_actuals).
    replay = json.loads((ROOT / config.REPORTS_DIR / "replay_2025-26.json").read_text())
    assert artifact["populations"]["min10"]["n"] == replay["n_restricted"]
    assert artifact["populations"]["all"]["n"] == replay["n_with_actuals"]
    for block in artifact["populations"].values():
        for tb in block["targets"].values():
            assert tb["coverage"] >= config.POLICY_MIN_COVERAGE


@pytest.mark.skipif(not REPORT.exists(), reason="policy artifact not written yet")
def test_site_summary_matches_committed_artifact() -> None:
    committed = json.loads((ROOT / evaluate.SITE_SUMMARY_PATH).read_text())
    expected = evaluate.site_summary_from_report(REPORT)
    assert committed == expected, "run: python -m nba.decisions.evaluate"
    assert committed["source_sha256"] == hashlib.sha256(REPORT.read_bytes()).hexdigest()
    assert committed["source_file"] == evaluate.report_path().as_posix()
    for block in committed["populations"].values():
        for tb in block["targets"].values():
            assert "coverage_curve" in tb and "verdict" in tb


def _slate() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2026-01-15"] * 3,
            "game_id": ["0022500900"] * 3,
            "player_id": [1, 2, 3],
            "player_name": ["A", "B", "C"],
            "team": ["LAL", "LAL", "BOS"],
            "opponent": ["BOS", "BOS", "LAL"],
            "home": [True, True, False],
            "pred_pts": [25.0, 10.0, 15.0],
            "pred_reb": [5.0, 5.0, 5.0],
            "pred_ast": [3.0, 3.0, 3.0],
            "pts_mean_last10": [20.0, 14.0, 15.0],
            "reb_mean_last10": [5.0, 5.0, 5.0],
            "ast_mean_last10": [3.0, 3.0, 3.0],
            "model_revision": ["m"] * 3,
            "dataset_revision": ["d"] * 3,
        }
    )


def test_decisions_apply_each_population_policy(tmp_path) -> None:
    artifact = policy.build_artifact(
        _frame(), "2025-26", generated_at="t", git_sha="s", input_description={}
    )
    for p in policy.POPULATIONS:
        artifact["populations"][p]["targets"]["pts"]["threshold"] = 2.0 if p == "min10" else 6.0
    files = decide.write_outputs(_slate(), date(2026, 1, 15), tmp_path / "decisions", artifact)
    assert [f.name for f in files] == ["2026-01-15.json", "latest.json"]
    payload = json.loads(files[1].read_text())
    assert payload["n_players"] == 3 and payload["policy"]["in_sample"] is True
    rows = {r["player_id"]: r for r in payload["rows"]}
    a = rows[1]["targets"]["pts"]
    assert a["edge"] == 5.0
    assert a["populations"]["min10"]["call"] == "over"
    assert a["populations"]["all"]["call"] == "no_call"
    assert rows[2]["targets"]["pts"]["populations"]["min10"]["call"] == "under"
    assert rows[3]["targets"]["pts"]["populations"]["min10"]["call"] == "no_call"
    q = artifact["populations"]["min10"]["targets"]["pts"]["bands"]["quantiles"]
    expected_band = [round(25 + q["q10"], 2), round(25 + q["q90"], 2)]
    assert a["populations"]["min10"]["band_80"] == expected_band
    assert payload["n_calls"]["min10"]["pts"] == 2 and payload["n_calls"]["all"]["pts"] == 0


def test_decide_cli_without_report_is_explicit(tmp_path, capsys) -> None:
    assert decide.main(["--date", "2026-01-15", "--report", str(tmp_path / "none.json")]) == 0
    assert capsys.readouterr().out.startswith("DECISIONS 2026-01-15: no policy report")
