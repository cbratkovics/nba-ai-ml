"""Drift: PSI bins, the day-aligned reference, the check, every policy branch, the streak,
the notifier, the calibration summary, the committed artifacts, and the nightly step."""

from __future__ import annotations

import json
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from nba import config
from nba.drift import calibrate, check, notify, policy, psi, reference, run
from nba.features import asof

ROOT = config.REPO_ROOT
REFERENCE = ROOT / reference.reference_path()
CALIBRATION = ROOT / calibrate.calibration_path()


# ---------- psi ----------


def test_psi_is_zero_on_identical_and_grows_with_a_shift() -> None:
    rng = np.random.default_rng(0)
    ref = pd.Series(rng.normal(0, 1, 5000))
    spec = psi.make_bins(ref)
    assert spec["kind"] == "quantile" and len(spec["edges"]) == 9
    expected = psi.proportions(ref, spec)
    assert abs(sum(expected) - 1) < 1e-9 and expected[-1] == 0.0
    assert psi.psi(ref, spec, expected) < 1e-9
    same = psi.psi(pd.Series(rng.normal(0, 1, 2000)), spec, expected)
    shifted = psi.psi(pd.Series(rng.normal(1, 1, 2000)), spec, expected)
    assert same < 0.05 < shifted


def test_psi_missing_bin_and_categorical() -> None:
    ref = pd.Series([0.0, 1.0] * 500)
    spec = psi.make_bins(ref)
    assert spec == {"kind": "categorical", "values": [0.0, 1.0]}
    expected = psi.proportions(ref, spec)
    assert expected == [0.5, 0.5, 0.0]
    # An unseen value counts as missing; NaN too.
    actual = psi.proportions(pd.Series([0.0, 1.0, 2.0, np.nan]), spec)
    assert actual == [0.25, 0.25, 0.5]
    assert psi.psi(pd.Series([0.0, 1.0, 2.0, np.nan]), spec, expected) > 0.5
    assert psi.proportions(pd.Series([], dtype="float64"), spec) == [0.0, 0.0, 0.0]


# ---------- reference ----------


def _gold_like(game_logs: pd.DataFrame) -> pd.DataFrame:
    rows = game_logs.copy()
    feats = check.population_rows(rows)
    flag = feats.set_index(["player_id", "game_id"])["in_population"]
    rows["population"] = np.where(
        flag.loc[list(zip(rows["player_id"], rows["game_id"], strict=True))].to_numpy(),
        "min10",
        "all",
    )
    rows["dataset_revision"] = "fixture"
    return rows


def test_reference_builds_from_gold_rows_and_validates(game_logs, monkeypatch) -> None:
    # The fixture has 2023-24 .. 2025-26; make the first two the training seasons.
    monkeypatch.setattr(config, "TRAIN_SEASONS", ("2023-24", "2024-25"))
    monkeypatch.setattr(config, "DRIFT_REFERENCE_SEASONS", ("2024-25",))
    monkeypatch.setattr(config, "DRIFT_MIN_ROWS", 5)
    ref = reference.build_reference(
        _gold_like(game_logs), generated_at="t", git_sha="s", source={"table": "fixture"}
    )
    assert ref["reference_seasons"] == ["2024-25"] and ref["seasons"] == ["2023-24", "2024-25"]
    assert set(ref["bins"]) == set(asof.FEATURE_COLUMNS)
    assert ref["all"]["n_rows"] > ref["reference_rows"] > 0
    assert list(ref["daily"])[0] == "day_001" and ref["max_season_day"] > 0
    first = ref["daily"]["day_001"]
    assert first["n_rows"] >= 0 and len(first["expected"]["pts_mean_last5"]) == (
        psi.n_bins(ref["bins"]["pts_mean_last5"]) + 1
    )
    problems = [p for p in reference.validate_reference(ref) if "season days" not in p]
    assert problems == []


# ---------- check ----------


def _season_dates() -> list[date]:
    start = date(2025, 10, 21)
    dates = [start + timedelta(days=i) for i in range(0, 174) if i not in range(115, 121)]
    return dates


def test_season_positions_and_reference_mode() -> None:
    dates = _season_dates()
    labels = check.season_positions(dates, "2025-26")
    assert labels[date(2025, 10, 21)] == "opening" and labels[date(2025, 10, 30)] == "opening"
    assert labels[date(2025, 11, 20)] == "cup"
    assert labels[date(2026, 2, 4)] == "deadline_week"
    assert labels[date(2026, 2, 19)] == "all_star_return"
    assert labels[date(2026, 4, 5)] == "april"
    assert labels[date(2026, 1, 10)] == "regular"
    ref = {"max_season_day": 170}
    assert check.reference_mode(date(2025, 10, 20), dates, ref) == "all"
    assert check.reference_mode(date(2025, 10, 21), dates, ref) == "all"
    assert check.reference_mode(date(2025, 11, 5), dates, ref) == "day_015"
    assert check.reference_mode(date(2026, 6, 1), dates, ref) == "day_171"
    assert check.season_of(date(2026, 3, 1)) == "2025-26"
    assert check.season_of(date(2025, 10, 1)) == "2025-26"


def test_check_reports_insufficient_window_without_psi(game_logs) -> None:
    feats = check.population_rows(game_logs)
    ref = json.loads(REFERENCE.read_text()) if REFERENCE.exists() else None
    if ref is None:
        pytest.skip("reference not written yet")
    result = check.check(feats, date(2025, 11, 20), ref)
    assert result["window"]["n_rows"] < config.DRIFT_MIN_ROWS and result["psi"] == {}
    assert result["window"]["start"] == "2025-11-06" and result["window"]["end"] == "2025-11-19"


# ---------- policy ----------


def _check(psi_values: dict[str, float], n_rows: int = 1000) -> dict:
    return {"window": {"n_rows": n_rows, "min_rows": config.DRIFT_MIN_ROWS}, "psi": psi_values}


CAL = {"chosen": {"threshold": 0.15, "min_features": 3}, "file": "reports/drift_calibration_x.json"}


def test_policy_every_branch() -> None:
    quiet = {"a": 0.01, "b": 0.02, "c": 0.03}
    noisy = {"a": 0.5, "b": 0.4, "c": 0.3, "d": 0.01}
    one = {"a": 0.5, "b": 0.01}
    # insufficient rows: no verdict on drift
    v = policy.decide(_check({}, n_rows=10), calibration=None, slate_status="ok", streak=0)
    assert v["status"] == "insufficient" and v["thresholds"]["calibrated"] is False
    # ok
    v = policy.decide(_check(quiet), calibration=CAL, slate_status="ok", streak=0)
    assert v["status"] == "ok" and v["flagged"] == [] and v["blocks_slate"] is False
    # warn: below the feature rule
    v = policy.decide(_check(one), calibration=CAL, slate_status="ok", streak=0)
    assert v["status"] == "warn" and v["flagged"] == ["a"]
    # warn: uncalibrated cannot hold
    v = policy.decide(_check(noisy), calibration=None, slate_status="ok", streak=0)
    assert v["status"] == "warn" and "uncalibrated" in v["reasons"][0]
    assert v["thresholds"]["psi"] == config.DRIFT_PROVISIONAL_THRESHOLD
    # hold: calibrated
    v = policy.decide(_check(noisy), calibration=CAL, slate_status="ok", streak=0)
    assert v["status"] == "hold" and v["flagged"] == ["a", "b", "c"] and v["blocks_slate"] is False
    assert v["thresholds"] == {
        "psi": 0.15,
        "min_features": 3,
        "calibrated": True,
        "calibration_file": "reports/drift_calibration_x.json",
    }
    # no-schedule streak: warn on an ok or insufficient run, never downgrades a hold
    n = config.NO_SCHEDULE_STREAK_WARN
    v = policy.decide(_check(quiet), calibration=CAL, slate_status="no_schedule", streak=n)
    assert v["status"] == "warn" and v["no_schedule_warn"] is True
    v = policy.decide(_check({}, 3), calibration=CAL, slate_status="no_schedule", streak=n)
    assert v["status"] == "warn"
    v = policy.decide(_check(noisy), calibration=CAL, slate_status="no_schedule", streak=n)
    assert v["status"] == "hold" and v["no_schedule_warn"] is True
    v = policy.decide(_check(quiet), calibration=CAL, slate_status="no_schedule", streak=n - 1)
    assert v["status"] == "ok" and v["no_schedule_warn"] is False


def test_no_schedule_streak_counts_consecutive_runs_newest_first() -> None:
    history = [
        {"date": "2026-10-01", "slate_status": "no_games"},
        {"date": "2026-10-02", "slate_status": "no_schedule"},
        {"date": "2026-10-03", "slate_status": "no_schedule"},
    ]
    assert policy.no_schedule_streak(history, "no_schedule") == 3
    assert policy.no_schedule_streak(history, "ok") == 0
    assert policy.no_schedule_streak([], "no_schedule") == 1
    broken = history + [{"date": "2026-10-04", "slate_status": "ok"}]
    assert policy.no_schedule_streak(broken, "no_schedule") == 1


# ---------- notify ----------


def test_notify_hold_and_streak(tmp_path, capsys) -> None:
    summary = {
        "date": "2026-11-01",
        "drift": {
            "status": "hold",
            "n_flagged": 4,
            "thresholds": {"psi": 0.15},
            "reasons": ["4 features at or above PSI 0.15"],
            "no_schedule_warn": True,
            "no_schedule_streak": 14,
        },
    }
    found = notify.issues(summary)
    assert [i["label"] for i in found] == [notify.HOLD_LABEL, notify.WARN_LABEL]
    assert "does not block" in found[0]["body"] and "14 consecutive" in found[1]["body"]
    assert notify.issues({"date": "x", "drift": {"status": "ok"}}) == []
    p = tmp_path / "s.json"
    p.write_text(json.dumps(summary))
    body = tmp_path / "body.md"
    assert notify.main(["--summary", str(p), "--body-out", str(body)]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0] == f"NOTIFY {notify.HOLD_LABEL} Nightly HOLD: feature drift"
    assert body.exists()
    assert notify.main(["--summary", str(tmp_path / "missing.json")]) == 0


# ---------- calibration ----------


def test_calibration_summary_chooses_smallest_zero_false_positive_threshold() -> None:
    dates = _season_dates()
    checks = []
    for i, d in enumerate(dates[:30]):
        position = "opening" if i < 10 else "cup"
        values = {f"f{k}": 0.02 for k in range(20)}
        if i == 15:  # one normal date with three features above 0.1 but below 0.15
            values.update({"f0": 0.12, "f1": 0.11, "f2": 0.14})
        checks.append(
            {
                "date": d.isoformat(),
                "position": position,
                "reference_mode": "day",
                "window": {"n_rows": 800 if i >= 3 else 10},
                "psi": values if i >= 3 else {},
            }
        )
    s = calibrate.summarise(checks, candidates=(0.05, 0.1, 0.15), min_features=3)
    assert s["chosen"]["threshold"] == 0.15 and s["chosen"]["false_positives"] == 0
    assert s["false_positives_total"] == {"0.05": 1, "0.1": 1, "0.15": 0}
    assert s["positions"]["cup"]["false_positives"]["0.1"]["dates"] == [dates[15].isoformat()]
    assert s["positions"]["opening"]["n_dates"] == 10
    assert s["positions"]["opening"]["n_dates_scored"] == 7
    regular = s["positions"]["regular"]
    assert regular["n_dates"] == 0 and regular["psi"]["max"] is None


@pytest.mark.skipif(not REFERENCE.exists(), reason="reference not written yet")
def test_committed_reference_is_keyed_and_valid() -> None:
    ref = json.loads(REFERENCE.read_text())
    assert reference.validate_reference(ref) == []
    expected_name = f"drift_reference_{asof.FEATURE_VERSION}_{config.MODEL_REVISION[:7]}.json"
    assert REFERENCE.name == expected_name
    assert ref["seasons"] == list(config.TRAIN_SEASONS)
    assert ref["reference_seasons"] == list(config.DRIFT_REFERENCE_SEASONS)
    assert ref["population"] == "min10" and ref["source"]["table"] == "gold.fct_player_game"
    assert ref["n_bins"] == config.DRIFT_BINS


@pytest.mark.skipif(not CALIBRATION.exists(), reason="calibration not written yet")
def test_committed_calibration_has_zero_false_positives_everywhere() -> None:
    cal = json.loads(CALIBRATION.read_text())
    assert calibrate.validate_calibration(cal) == []
    assert cal["season"] == config.HOLDOUT_SEASON and cal["n_dates"] == 164
    chosen = str(cal["chosen"]["threshold"])
    assert cal["chosen"]["false_positives"] == 0
    for pos in check.POSITIONS:
        assert cal["positions"][pos]["false_positives"][chosen]["count"] == 0
    # The calibration is what lets the policy HOLD.
    loaded = calibrate.load_calibration()
    th = policy.thresholds(loaded)
    assert th["calibrated"] is True and th["psi"] == cal["chosen"]["threshold"]
    assert th["calibration_file"] == calibrate.calibration_path().as_posix()


# ---------- the nightly step ----------


@pytest.mark.skipif(not REFERENCE.exists(), reason="reference not written yet")
def test_run_writes_a_report_and_counts_the_streak(tmp_path, game_logs) -> None:
    root = tmp_path
    (root / "drift").mkdir()
    for i in range(1, 4):
        d = date(2026, 10, 1) + timedelta(days=i)
        (root / "drift" / f"{d.isoformat()}.json").write_text(
            json.dumps({"date": d.isoformat(), "slate_status": "no_schedule"})
        )
    report, path = run.run(root, date(2026, 10, 5), game_logs, "no_schedule")
    assert path == root / "drift" / "2026-10-05.json" and path.exists()
    assert report["status"] == "insufficient"  # fixture logs: far fewer than 500 rows
    assert report["no_schedule_streak"] == 4 and report["no_schedule_warn"] is False
    assert report["reference_mode"] == "all" and report["position"] == "off_season"
    assert report["features"] == [] and report["blocks_slate"] is False
    assert report["thresholds"]["calibrated"] == CALIBRATION.exists()
    assert "DRIFT 2026-10-05: status=insufficient" in run.log_line(date(2026, 10, 5), report)
