"""Season readiness: the first 2026-27 nights, end to end on fixtures (docs/AE_AUDIT.md §1.8).

Night 1 (2026-10-20) has a schedule file for a season no stored row belongs to yet: the slate
is written and labelled 2026-27, the warehouse gate says build, and a `--selector nightly`
build over a warehouse that holds only earlier seasons accepts the new rows (the seasons seed
lists 2026-27, the slv_game_logs contract holds, the bronze product families switch from their
empty typed relations to real files). Night 2 scores the residuals, the incremental build picks
them up, and a restatement inside the lookback is applied on the next incremental run. The
export and a dry-run publish close the path the nightly job runs (steps 11 to 14).

Same dbt harness as tests/test_dbt_incremental.py; the singular reconciliation tests are
excluded because the fixture rows are not the published replay.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import date
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from nba import config, nightly, rehearsal
from nba.ingest import kaggle_dump, schedule
from nba.storage import hf, local
from nba.warehouse import publish
from tests.conftest import TEAMS
from tests.test_dbt_gold import _fixture_replay_rows
from tests.test_dbt_incremental import _write_root
from tests.test_slate import fake_models

pytest.importorskip("dbt.cli.main")

OPENING = date(2026, 10, 20)
GAMES = [
    {
        "gameId": 22600001,
        "gameDateTimeEst": "2026-10-20 19:30:00",
        "homeTeamId": TEAMS[0][0],
        "awayTeamId": TEAMS[1][0],
    },
    {
        "gameId": 22600002,
        "gameDateTimeEst": "2026-10-20 22:00:00",
        "homeTeamId": TEAMS[2][0],
        "awayTeamId": TEAMS[3][0],
    },
]
# The nightly selector is the whole project (dbt/selectors.yml); `--exclude` is ignored next to
# `--selector`, so the test spells the same selection out and drops the singular tests, which
# compare against the published replay the fixture rows are not.
NIGHTLY_WITHOUT_SINGULAR = ("--select", "fqn:*", "--exclude", "test_type:singular")
SAFE_SINGULAR_TESTS = (
    "assert_season_game_counts",
    "assert_lookback_covers_restatement_lag",
    "assert_snapshot_current_matches_stints",
    "assert_known_missing_games_are_absent",
)


def _dbt(db: Path, warehouse_root: Path, *args: str) -> dict[str, str]:
    script = Path(sys.executable).parent / "dbt"
    cmd = [
        *([str(script)] if script.exists() else [sys.executable, "-m", "dbt"]),
        *args,
        "--project-dir",
        "dbt",
        "--profiles-dir",
        "dbt",
        "--target",
        "local",
        "--target-path",
        str(db.parent / "target"),
        "--log-path",
        str(db.parent / "logs"),
        "--vars",
        json.dumps({"warehouse_root": str(warehouse_root)}),
    ]
    env = {**os.environ, "NBA_DUCKDB_PATH": str(db), "DBT_TARGET": "local"}
    proc = subprocess.run(cmd, cwd=config.REPO_ROOT, env=env, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-3000:]
    results = json.loads((db.parent / "target" / "run_results.json").read_text())
    return {r["unique_id"].split(".")[2]: r["status"] for r in results["results"]}


def _query(db: Path, sql: str):
    con = duckdb.connect(str(db), read_only=True)
    try:
        return con.execute(sql).fetchall()
    finally:
        con.close()


def _dump(kaggle_dir: Path, tmp_path: Path, games: list[dict]) -> Path:
    d = tmp_path / "dump"
    d.mkdir()
    for name in (kaggle_dump.BOX_SCORE_FILE, kaggle_dump.TEAM_HISTORY_FILE):
        (d / name).write_bytes((kaggle_dir / name).read_bytes())
    pd.DataFrame(games).to_csv(d / schedule.schedule_file_name("2026-27"), index=False)
    return d


def _actual_rows(predictions: pd.DataFrame, logs: pd.DataFrame) -> pd.DataFrame:
    """A box score for every slated player, copied from their latest game, dated opening night."""
    latest = logs.sort_values("game_date").groupby("player_id").tail(1).set_index("player_id")
    rows = latest.loc[predictions["player_id"].to_numpy()].reset_index()
    rows["game_id"] = predictions["game_id"].to_numpy()
    rows["game_date"] = pd.Timestamp(OPENING)
    rows["season"] = "2026-27"
    rows["team"] = predictions["team"].to_numpy()
    rows["opponent"] = predictions["opponent"].to_numpy()
    rows["home"] = predictions["home"].to_numpy().astype(bool)
    rows["minutes"] = 30.0  # every slated player plays enough to join the training population
    return rows[list(logs.columns)]


def test_opening_nights_end_to_end(
    game_logs: pd.DataFrame, kaggle_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    local.write_per_season(game_logs, root / config.DATA_DIR)
    for name in ("pull_dataset", "pull_products", "push_dataset", "push_products", "push_gold"):
        monkeypatch.setattr(
            hf, name, lambda *a, _n=name, **k: pytest.fail(f"{_n} must not be called")
        )
    monkeypatch.setattr(config, "GROQ_API_KEY", None)
    dump = _dump(kaggle_dir, tmp_path, GAMES)
    assert not (game_logs["season"] == "2026-27").any(), "the fixture must predate the season"

    # Night 1: a season label no stored row carries yet.
    s1 = nightly.run(
        OPENING,
        push=False,
        root=root,
        local_dump=dump,
        report_path=tmp_path / "r1.json",
        load_models=fake_models,
    )
    assert s1.slate["status"] == "ok" and s1.slate["n_games"] == 2 and s1.slate["n_players"] == 8
    assert s1.decisions["status"] == "ok", "the committed policy artifact applies to the slate"
    assert s1.drift["status"] == "insufficient" and s1.drift["no_schedule_streak"] == 0
    predictions = pd.read_parquet(root / "predictions" / f"{OPENING}.parquet")
    assert set(predictions["date"]) == {OPENING.isoformat()}
    from nba.warehouse import gate

    assert gate.decide(json.loads(json.dumps(s1.__dict__, default=str)))[0]

    wh = _write_root(game_logs, tmp_path / "wh")
    replay_rows, _ = _fixture_replay_rows(game_logs)
    folder = wh / "replay" / config.HOLDOUT_SEASON / "residuals"
    folder.mkdir(parents=True)
    replay_rows.to_parquet(folder / "fixture.parquet", index=False)
    rehearsal.stage_products(root, wh)
    db = tmp_path / "w" / "nba.duckdb"
    db.parent.mkdir()
    statuses = _dbt(db, wh, "build", *NIGHTLY_WITHOUT_SINGULAR, "--full-refresh")
    assert set(statuses.values()) <= {"success", "pass", "warn"}, statuses
    assert statuses["brz_predictions"] == "success" and statuses["slv_game_logs"] == "success"
    assert _query(
        db,
        "select season, count(*) from silver.slv_predictions where run_kind = 'nightly' group by 1",
    ) == [("2026-27", 8)]
    assert [
        s for n, s in statuses.items() if n.startswith("relationships_slv_predictions_season")
    ] == ["pass"]
    assert _query(
        db,
        "select count(*) from gold.mart_drift "
        "where run_date = date '2026-10-20' and status = 'insufficient'",
    ) == [(1,)]

    # Night 2: the games have been played; residuals score the slate and the build is incremental.
    logs = local.read_game_logs(root / config.DATA_DIR)
    merged = pd.concat([logs, _actual_rows(predictions, logs)], ignore_index=True)
    local.write_per_season(merged, root / config.DATA_DIR)
    s2 = nightly.run(
        date(2026, 10, 21),
        push=False,
        root=root,
        local_dump=dump,
        report_path=tmp_path / "r2.json",
        load_models=fake_models,
    )
    assert s2.residuals["n_predicted"] == 8 and s2.residuals["n_with_actuals"] == 8
    assert s2.slate["status"] == "no_games"
    _write_root(merged, wh)
    rehearsal.stage_products(root, wh)
    statuses = _dbt(db, wh, "build", *NIGHTLY_WITHOUT_SINGULAR)
    assert set(statuses.values()) <= {"success", "pass", "warn"}, statuses
    assert _query(
        db, "select count(*) from gold.fct_prediction where run_kind = 'nightly' and has_actual"
    ) == [(8,)]
    assert _query(
        db, "select count(*), max(population) from gold.fct_player_game where season = '2026-27'"
    ) == [(8, "all")]  # opening night: every row is a season debut, so no min10 row yet
    assert _query(
        db,
        "select count(*) from gold.mart_daily_metrics where run_kind = 'nightly' "
        "and game_date = date '2026-10-20' and population = 'all'",
    ) == [(3,)]
    assert _query(
        db,
        "select population, count(*) from gold.mart_policy_metrics "
        "where run_kind = 'nightly' and threshold is not null group by 1",
    ) == [("all", 3)]  # the min10 policy has no resolved rows until players have a prior game
    assert _query(db, "select count(*) from gold.mart_drift where season = '2026-27'") == [(2,)]
    assert _query(
        db, "select report_date, within_lookback from gold.mart_restatement_lag order by 1"
    ) == [(date(2026, 10, 20), True), (date(2026, 10, 21), True)]
    assert _query(
        db, "select games, meets_contract from gold.mart_season_coverage where season = '2026-27'"
    ) == [(2, False)]
    # The singular tests that do not compare against the published replay still hold: the
    # new season is exempt from the game-count contract until it is 150 days old.
    statuses = _dbt(db, wh, "test", "--select", *SAFE_SINGULAR_TESTS)
    assert set(statuses.values()) == {"pass"}, statuses

    # A restatement inside the lookback reaches the gold fact on the next incremental run.
    pid, gid = int(predictions.iloc[0]["player_id"]), str(predictions.iloc[0]["game_id"])
    before = _query(
        db, f"select pts from gold.fct_player_game where player_id = {pid} and game_id = '{gid}'"
    )[0][0]
    restated = merged.copy()
    mask = (restated["player_id"] == pid) & (restated["game_id"] == gid)
    restated.loc[mask, "pts"] = restated.loc[mask, "pts"] + 7
    _write_root(restated, wh)
    _dbt(db, wh, "build", *NIGHTLY_WITHOUT_SINGULAR)
    after = _query(
        db, f"select pts from gold.fct_player_game where player_id = {pid} and game_id = '{gid}'"
    )[0][0]
    assert after == before + 7

    # Steps 13 and 14 of the nightly job: export, then (dry-run) publish.
    out_dir = tmp_path / "export"
    out_dir.mkdir()
    _dbt(db, wh, "run-operation", "export_gold", "--args", json.dumps({"out_dir": str(out_dir)}))
    manifest = json.loads((out_dir / "_export_manifest.json").read_text())
    # Exactly the gold models without meta export=false are exported; the raw-lines tables
    # (fct_paper_bet, slv_lines, brz_lines) must never be among them (ADR-0031).
    dbt_manifest = json.loads((db.parent / "target" / "manifest.json").read_text())
    exported_expected = {
        n["alias"]
        for n in dbt_manifest["nodes"].values()
        if n["resource_type"] == "model"
        and "gold" in n["tags"]
        and (n["config"].get("meta") or {}).get("export", True) is not False
    }
    exported = {m["model"] for m in manifest}
    assert exported == exported_expected, exported ^ exported_expected
    assert exported.isdisjoint({"fct_paper_bet", "brz_lines", "slv_lines"})
    assert exported >= {"fct_prediction", "mart_policy_metrics", "mart_drift", "mart_ledger"}
    assert publish.publish(out_dir, dry_run=True) is None
    assert len(publish.export_files(out_dir)) == len(exported) + 1  # plus the manifest
