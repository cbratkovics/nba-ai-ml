"""The feature parity proof (ADR-0032), three ways:

1. The committed evidence report reports/feature_parity_asof_v1.json names exactly the feature
   module's 23 features, joined every row on both sides, saw no null mismatch, and the dbt
   tolerance var covers every observed difference (with the integer features exact).
2. The SQL macro's feature list is the Python module's, in order.
3. On fixture rows, the dbt singular test passes against the module's output and fails on
   exactly one row when one value of that output is perturbed (the proof the test can fail).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from nba import config
from nba.features import asof
from nba.warehouse import features, parity

REPORT = parity.report_path()
PARITY_TEST = "assert_int_features_match_feature_module"


def _vars() -> dict:
    return yaml.safe_load((config.REPO_ROOT / "dbt" / "dbt_project.yml").read_text())["vars"]


def test_report_covers_every_feature_and_the_tolerance_covers_every_difference() -> None:
    report = json.loads(REPORT.read_text())
    assert report["feature_version"] == asof.FEATURE_VERSION
    assert list(report["features"]) == asof.FEATURE_COLUMNS
    assert report["rows_sql"] == report["rows_module"] == report["rows_joined"] > 100_000
    assert report["total_null_mismatches"] == 0
    tol = float(_vars()["tol_feature_parity"])
    assert report["tolerance"] == tol, "re-run python -m nba.warehouse.parity after a var change"
    for name, f in report["features"].items():
        assert f["null_mismatches"] == 0, name
        if f["kind"] == "integer":
            assert f["max_abs_diff"] == 0, name
        else:
            assert f["max_abs_diff"] <= tol, (name, f["max_abs_diff"], tol)
    # The tolerance is set from the evidence, not the other way round: it stays within two
    # orders of magnitude of the largest observed difference and far below a formula change.
    assert report["max_abs_diff_doubles"] <= tol <= max(1e-12, report["max_abs_diff_doubles"] * 100)
    assert tol < 1e-6


def test_sql_feature_list_is_the_modules() -> None:
    macro = (config.REPO_ROOT / "dbt" / "macros" / "feature_columns.sql").read_text()
    # The macro builds the names from the same stat/window loops; render them the same way.
    names = [f"{s}_mean_last{w}" for s in ("pts", "reb", "ast", "minutes") for w in (5, 10, 20)]
    names += [f"{s}_mean_season" for s in ("pts", "reb", "ast", "minutes")]
    names += ["games_played_season", "days_rest", "back_to_back", "home"]
    names += [f"{s}_mean_vs_opp" for s in ("pts", "reb", "ast")]
    assert names == asof.FEATURE_COLUMNS
    assert "['pts', 'reb', 'ast', 'minutes']" in macro and "[5, 10, 20]" in macro
    assert set(parity.INTEGER_FEATURES) == {"games_played_season", "back_to_back", "home"}
    for name in parity.INTEGER_FEATURES:
        assert f"'{name}'" in macro


def test_write_features_emits_ids_features_and_version(tmp_path: Path, game_logs) -> None:
    from nba.storage import local

    local.write_per_season(game_logs, tmp_path / config.HF_DATASET_PREFIX)
    path, n = features.write_features(tmp_path)
    assert path == tmp_path / "features" / f"{asof.FEATURE_VERSION}.parquet"
    out = pd.read_parquet(path)
    assert n == len(out) == len(game_logs)
    assert list(out.columns) == list(asof.ID_COLUMNS) + asof.FEATURE_COLUMNS + ["feature_version"]
    assert (out["feature_version"] == asof.FEATURE_VERSION).all()


def test_dbt_parity_test_passes_on_fixture_rows_and_fails_on_a_perturbed_value(
    tmp_path: Path, game_logs
) -> None:
    pytest.importorskip("dbt.cli.main")
    from tests.test_dbt_gold import _dbt
    from tests.test_dbt_incremental import _write_root

    root = _write_root(game_logs, tmp_path / "wh")
    db = tmp_path / "db" / "w.duckdb"
    db.parent.mkdir()
    # The model's ancestors (bronze game logs, the manifest, slv_game_logs), the module side
    # and the parity test; cautious indirect selection keeps every other test out.
    select = ("+int_player_game_features", "brz_features_asof_v1", PARITY_TEST)
    status = _dbt(db, root, tmp_path / "reports", *select)
    assert status[PARITY_TEST] == "pass", status

    path = features.features_path(root)
    frame = pd.read_parquet(path)
    row = frame["pts_mean_last5"].first_valid_index()
    frame.loc[row, "pts_mean_last5"] += 0.01
    frame.to_parquet(path, index=False)
    with pytest.raises(AssertionError) as info:
        _dbt(db, root, tmp_path / "reports", *select)
    text = str(info.value)
    assert PARITY_TEST in text and "Got 1 result" in text, text[-1500:]
