"""Gold-layer proofs on fixture data, no network:

1. fct_player_game.<stat>_mean_season_prior (the decision policy's second causal baseline)
   equals the feature module's <stat>_mean_season row for row.
2. The policy marts reproduce a policy artifact built by nba/decisions/policy.py from the
   same rows: fixture replay residuals go into a scratch warehouse root, the artifact into a
   scratch reports root, and the dbt singular tests assert_policy_* run inside `dbt build`
   (cautious indirect selection keeps every other reconciliation test out). SQL and pandas
   are two implementations of the same rule; this is the proof they agree.
3. Without an artifact the policy marts are empty, not broken.

Same harness as tests/test_dbt_incremental.py.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nba import config
from nba.decisions import policy
from nba.features import asof
from nba.predict import residuals as residuals_module
from tests.test_dbt_incremental import _write_root

pytest.importorskip("dbt.cli.main")
duckdb = pytest.importorskip("duckdb")

POLICY_TESTS = (
    "assert_policy_metrics_reconcile_to_policy_report",
    "assert_policy_sweep_reconciles_to_policy_report",
)


def _dbt(db_path: Path, warehouse_root: Path, reports_root: Path, *select: str) -> dict:
    script = Path(sys.executable).parent / "dbt"
    cmd = [
        *([str(script)] if script.exists() else [sys.executable, "-m", "dbt"]),
        "build",
        "--select",
        *select,
        "--indirect-selection",
        "cautious",
        "--project-dir",
        "dbt",
        "--profiles-dir",
        "dbt",
        "--target",
        "local",
        "--target-path",
        str(db_path.parent / "target"),
        "--log-path",
        str(db_path.parent / "logs"),
        "--full-refresh",
        "--vars",
        json.dumps({"warehouse_root": str(warehouse_root), "reports_root": str(reports_root)}),
    ]
    env = {**os.environ, "NBA_DUCKDB_PATH": str(db_path), "DBT_TARGET": "local"}
    proc = subprocess.run(cmd, cwd=config.REPO_ROOT, env=env, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-3000:]
    results = json.loads((db_path.parent / "target" / "run_results.json").read_text())
    return {r["unique_id"].split(".")[-1]: r["status"] for r in results["results"]}


def _fixture_replay_rows(game_logs: pd.DataFrame) -> pd.DataFrame:
    """Replay residual rows for the holdout season in the nightly residual layout, with a
    deterministic prediction rule so the calls are non-trivial."""
    feats = asof.build_features(game_logs)
    rows = game_logs.merge(
        feats[
            ["player_id", "game_id"]
            + [f"{t}_mean_last10" for t in config.TARGETS]
            + [f"{t}_mean_season" for t in config.TARGETS]
        ],
        on=["player_id", "game_id"],
    )
    rows = rows[rows["season"] == config.HOLDOUT_SEASON].reset_index(drop=True)
    rng = np.random.default_rng(7)
    out = pd.DataFrame(
        {
            "date": rows["game_date"].dt.date,
            "game_id": rows["game_id"],
            "player_id": rows["player_id"],
            "player_name": rows["player_name"],
            "team": rows["team"],
            "opponent": rows["opponent"],
            "home": rows["home"].astype(bool),
            "minutes": rows["minutes"].astype(float),
            "has_actual": True,
            "game_ingested": True,
            "model_revision": "fixture-model",
            "dataset_revision": "fixture",
        }
    )
    for t in config.TARGETS:
        last10 = rows[f"{t}_mean_last10"]
        pred = last10 + rng.normal(0, 2.0, len(rows)) + 0.3 * (rows[t] - last10)
        out[f"pred_{t}"] = pred.fillna(rows[t].astype(float))
        out[f"actual_{t}"] = rows[t].astype(float)
        out[f"resid_{t}"] = out[f"actual_{t}"] - out[f"pred_{t}"]
        out[f"{t}_mean_last10"] = last10
    both = pd.Series(True, index=rows.index)
    for t in config.TARGETS:
        both &= rows[f"{t}_mean_last10"].notna() & rows[f"{t}_mean_season"].notna()
    out["in_metrics_population"] = (out["minutes"] >= config.MIN_MINUTES) & both
    out = out[list(residuals_module.RESIDUAL_COLUMNS)]
    season = {t: rows[f"{t}_mean_season"] for t in config.TARGETS}
    return out, season


def _policy_input(res: pd.DataFrame, season: dict[str, pd.Series]) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "player_id": res["player_id"],
            "game_id": res["game_id"],
            "has_actual": res["has_actual"],
            "baseline_defined": pd.concat(
                [res[f"{t}_mean_last10"].notna() for t in config.TARGETS], axis=1
            ).all(axis=1),
            "in_metrics_population": res["in_metrics_population"],
            "model_revision": res["model_revision"],
            "feature_version": asof.FEATURE_VERSION,
            "dataset_revision": res["dataset_revision"],
        }
    )
    for t in config.TARGETS:
        frame[f"pred_{t}"] = res[f"pred_{t}"]
        frame[f"baseline_last10_{t}"] = res[f"{t}_mean_last10"]
        frame[f"baseline_season_{t}"] = season[t]
        frame[f"actual_{t}"] = res[f"actual_{t}"]
    return frame


@pytest.fixture
def warehouse(tmp_path, game_logs):
    root = _write_root(game_logs, tmp_path / "wh")
    res, season = _fixture_replay_rows(game_logs)
    folder = root / "replay" / config.HOLDOUT_SEASON / "residuals"
    folder.mkdir(parents=True)
    res.to_parquet(folder / "fixture.parquet", index=False)
    reports = tmp_path / "reports"
    reports.mkdir()
    return root, reports, res, season


def test_policy_marts_are_empty_without_an_artifact(tmp_path, warehouse) -> None:
    root, reports, _, _ = warehouse
    db = tmp_path / "nba.duckdb"
    statuses = _dbt(db, root, reports, "+mart_policy_metrics", "+mart_policy_sweep")
    assert all(s in ("success", "pass") for s in statuses.values()), statuses
    con = duckdb.connect(str(db), read_only=True)
    try:
        assert con.execute("select count(*) from gold.fct_decision_policy").fetchone()[0] > 0
        assert (
            con.execute(
                "select count(*) from gold.fct_decision_policy where decision is not null"
            ).fetchone()[0]
            == 0
        )
        assert con.execute("select count(*) from gold.mart_policy_sweep").fetchone()[0] == 0
        n_metrics = con.execute("select count(*) from gold.mart_policy_metrics").fetchone()[0]
        assert n_metrics == 6  # both populations x three targets, thresholds null
    finally:
        con.close()


def test_sql_policy_reproduces_the_pandas_artifact_and_the_season_mean(
    tmp_path, game_logs, warehouse
) -> None:
    root, reports, res, season = warehouse
    artifact = policy.build_artifact(
        _policy_input(res, season),
        config.HOLDOUT_SEASON,
        generated_at="2026-01-01T00:00:00+00:00",
        git_sha="fixture",
        input_description={"source": "fixture"},
    )
    assert policy.validate_artifact(artifact) == []
    (reports / f"policy_{config.HOLDOUT_SEASON}.json").write_text(json.dumps(artifact))
    db = tmp_path / "nba.duckdb"
    statuses = _dbt(db, root, reports, "+mart_policy_metrics", "+mart_policy_sweep")
    for name in POLICY_TESTS:
        assert statuses.get(name) == "pass", (name, statuses)
    con = duckdb.connect(str(db), read_only=True)
    try:
        got = con.execute(
            "select player_id, game_id, pts_mean_season_prior, reb_mean_season_prior, "
            "ast_mean_season_prior from gold.fct_player_game"
        ).df()
        decided = con.execute(
            "select population, target, decision, count(*) as n from gold.fct_decision_policy "
            "group by 1, 2, 3 order by 1, 2, 3"
        ).df()
    finally:
        con.close()
    assert set(decided["decision"]) == {"over", "under", "no_call"}
    feats = asof.build_features(game_logs)[
        ["player_id", "game_id", "pts_mean_season", "reb_mean_season", "ast_mean_season"]
    ]
    merged = got.merge(feats, on=["player_id", "game_id"], how="outer", indicator=True)
    assert (merged["_merge"] == "both").all() and len(merged) == len(game_logs)
    for stat in config.TARGETS:
        sql, py = merged[f"{stat}_mean_season_prior"], merged[f"{stat}_mean_season"]
        assert (sql.isna() == py.isna()).all(), f"{stat}: null pattern differs"
        np.testing.assert_allclose(sql.dropna(), py.dropna(), rtol=0, atol=1e-9)
