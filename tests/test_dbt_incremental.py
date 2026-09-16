"""Incremental equivalence for slv_game_logs: a full refresh and an incremental run over the
same input produce identical rows; a restated game inside the lookback is picked up; one
outside it is not until --full-refresh. Runs the dbt console script next to the interpreter
against a scratch DuckDB file and a scratch warehouse root holding fixture game logs.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from nba import config
from nba.storage import local

pytest.importorskip("dbt.cli.main")
duckdb = pytest.importorskip("duckdb")

MODEL = "slv_game_logs"
KEYS = ["player_id", "game_id"]
LOOKBACK_DAYS = 14
DBT_DIR = config.REPO_ROOT / "dbt"


def _dbt_bin() -> list[str]:
    script = Path(sys.executable).parent / "dbt"
    return [str(script)] if script.exists() else [sys.executable, "-m", "dbt"]


def _dbt_run(db_path: Path, warehouse_root: Path, *, full_refresh: bool) -> None:
    cmd = [
        *_dbt_bin(),
        "run",
        "--select",
        f"+{MODEL}",
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
        "--vars",
        json.dumps({"warehouse_root": str(warehouse_root), "lookback_days": LOOKBACK_DAYS}),
    ]
    if full_refresh:
        cmd.append("--full-refresh")
    env = {**os.environ, "NBA_DUCKDB_PATH": str(db_path), "DBT_TARGET": "local"}
    proc = subprocess.run(cmd, cwd=config.REPO_ROOT, env=env, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]


def _table(db_path: Path) -> pd.DataFrame:
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        return (
            con.execute(f"select * from silver.{MODEL}")
            .df()
            .sort_values(KEYS)
            .reset_index(drop=True)
        )
    finally:
        con.close()


def _checksum(df: pd.DataFrame) -> str:
    d = df[sorted(df.columns)].sort_values(KEYS).reset_index(drop=True)
    return hashlib.sha256(
        pd.util.hash_pandas_object(d, index=False).to_numpy().tobytes()
    ).hexdigest()


def _write_root(frame: pd.DataFrame, root: Path) -> Path:
    """A warehouse root with per-season parquet and a manifest, as the loader lays it out."""
    from nba.warehouse import load

    if root.exists():
        for p in (root / config.HF_DATASET_PREFIX).glob("*.parquet"):
            p.unlink()
    local.write_per_season(frame, root / config.HF_DATASET_PREFIX)
    load.write_manifest(
        root, "fixture", config.HOLDOUT_SEASON, loaded_at="2026-01-01T00:00:00+00:00"
    )
    return root


def _restate(frame: pd.DataFrame, game_date: pd.Timestamp) -> pd.DataFrame:
    out = frame.copy()
    idx = out[out["game_date"] == game_date].index[0]
    out.loc[idx, "pts"] = int(out.loc[idx, "pts"]) + 7
    return out


@pytest.fixture
def dates(game_logs: pd.DataFrame) -> list[pd.Timestamp]:
    return sorted(game_logs["game_date"].unique())


def test_incremental_run_equals_full_refresh_after_append(tmp_path, game_logs, dates) -> None:
    # One warehouse root per scenario: source_file records the parquet path, so the files
    # must live at the same path for the two builds to be comparable.
    root = tmp_path / "wh"
    cutoff = dates[-3]
    _write_root(game_logs, root)
    ref_db = tmp_path / "ref" / "w.duckdb"
    ref_db.parent.mkdir()
    _dbt_run(ref_db, root, full_refresh=True)
    reference = _table(ref_db)
    _write_root(game_logs[game_logs["game_date"] < cutoff], root)
    inc_db = tmp_path / "inc" / "w.duckdb"
    inc_db.parent.mkdir()
    _dbt_run(inc_db, root, full_refresh=True)
    assert len(_table(inc_db)) < len(reference)
    _write_root(game_logs, root)
    _dbt_run(inc_db, root, full_refresh=False)
    after = _table(inc_db)
    assert len(after) == len(reference) and _checksum(after) == _checksum(reference)


def test_restated_game_inside_lookback_is_picked_up(tmp_path, game_logs, dates) -> None:
    root = tmp_path / "wh"
    inside = [d for d in dates if (dates[-1] - d).days <= LOOKBACK_DAYS][0]
    _write_root(game_logs, root)
    db = tmp_path / "w.duckdb"
    _dbt_run(db, root, full_refresh=True)
    _write_root(_restate(game_logs, inside), root)
    _dbt_run(db, root, full_refresh=False)
    ref_db = tmp_path / "ref.duckdb"
    _dbt_run(ref_db, root, full_refresh=True)
    assert _checksum(_table(db)) == _checksum(_table(ref_db))


def test_restated_game_outside_lookback_needs_full_refresh(tmp_path, game_logs, dates) -> None:
    root = tmp_path / "wh"
    outside = [d for d in dates if (dates[-1] - d).days > LOOKBACK_DAYS + 1][-1]
    _write_root(game_logs, root)
    db = tmp_path / "w.duckdb"
    _dbt_run(db, root, full_refresh=True)
    original = _table(db)
    _write_root(_restate(game_logs, outside), root)
    _dbt_run(db, root, full_refresh=False)
    assert _checksum(_table(db)) == _checksum(original)
    _dbt_run(db, root, full_refresh=True)
    assert _checksum(_table(db)) != _checksum(original)
