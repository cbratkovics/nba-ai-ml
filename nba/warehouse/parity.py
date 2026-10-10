"""Measure the feature parity between int_player_game_features and the feature module.

The dbt singular test assert_int_features_match_feature_module decides pass or fail; this
script records the evidence behind its tolerance (ADR-0032): for every one of the 23 asof_v1
features, the largest absolute difference observed between the SQL recomputation and the
Python module over the whole warehouse, the number of rows where one side is null and the
other is not, and the row counts on each side. Written to reports/feature_parity_<version>.json;
tests/test_feature_parity.py checks that the dbt var tol_feature_parity covers every observed
difference and that the feature list is the module's.

    python -m nba.warehouse.parity [--duckdb .duckdb/nba.duckdb] [--out reports/...]
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from nba import config
from nba.features import asof

DEFAULT_DUCKDB = Path(".duckdb") / "nba.duckdb"
INTEGER_FEATURES: tuple[str, ...] = ("games_played_season", "back_to_back", "home")
SQL_TABLE = "intermediate.int_player_game_features"
MODULE_TABLE = "bronze.brz_features_asof_v1"


def report_path(version: str = asof.FEATURE_VERSION) -> Path:
    return config.REPO_ROOT / "reports" / f"feature_parity_{version}.json"


def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return out or None


def measure(con, features: list[str] = asof.FEATURE_COLUMNS) -> dict:
    """Per-feature max |SQL - module| and null mismatches over the inner join on the grain,
    plus the counts that prove the join covered both sides."""
    join = f"from {SQL_TABLE} as s join {MODULE_TABLE} as m using (player_id, game_id)"
    exprs = []
    for f in features:
        exprs.append(f"max(abs(s.{f} - m.{f})) as max_{f}")
        exprs.append(f"count(*) filter (where (s.{f} is null) <> (m.{f} is null)) as nulls_{f}")
    row = con.execute(f"select {', '.join(exprs)} {join}").fetchone()
    n_sql = con.execute(f"select count(*) from {SQL_TABLE}").fetchone()[0]
    n_module = con.execute(f"select count(*) from {MODULE_TABLE}").fetchone()[0]
    n_joined = con.execute(f"select count(*) {join}").fetchone()[0]
    per_feature = {}
    for i, f in enumerate(features):
        max_diff, nulls = row[2 * i], row[2 * i + 1]
        per_feature[f] = {
            "kind": "integer" if f in INTEGER_FEATURES else "double",
            "max_abs_diff": None if max_diff is None else float(max_diff),
            "null_mismatches": int(nulls),
        }
    doubles = [
        v["max_abs_diff"]
        for f, v in per_feature.items()
        if v["kind"] == "double" and v["max_abs_diff"] is not None
    ]
    return {
        "rows_sql": int(n_sql),
        "rows_module": int(n_module),
        "rows_joined": int(n_joined),
        "features": per_feature,
        "max_abs_diff_doubles": max(doubles) if doubles else None,
        "max_abs_diff_integers": max(
            (v["max_abs_diff"] or 0.0) for v in per_feature.values() if v["kind"] == "integer"
        ),
        "total_null_mismatches": sum(v["null_mismatches"] for v in per_feature.values()),
    }


def write_report(duckdb_path: Path, out: Path, tolerance: float) -> dict:
    import duckdb

    con = duckdb.connect(str(duckdb_path), read_only=True)
    try:
        measured = measure(con)
        revision = con.execute("select max(dataset_revision) from silver.slv_game_logs").fetchone()[
            0
        ]
    finally:
        con.close()
    report = {
        "feature_version": asof.FEATURE_VERSION,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
        "duckdb": str(duckdb_path),
        "dataset_revision": revision,
        "sql_table": SQL_TABLE,
        "module_table": MODULE_TABLE,
        "tolerance": tolerance,
        "tolerance_note": (
            "dbt var tol_feature_parity, applied to the double features by "
            "assert_int_features_match_feature_module; integer features must match exactly. "
            "Set from max_abs_diff_doubles with headroom for summation order; a formula "
            "difference would show as at least 1e-3."
        ),
        **measured,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    return report


def dbt_tolerance() -> float:
    import yaml

    project = yaml.safe_load((config.REPO_ROOT / "dbt" / "dbt_project.yml").read_text())
    return float(project["vars"]["tol_feature_parity"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--duckdb", type=Path, default=DEFAULT_DUCKDB)
    parser.add_argument("--out", type=Path, default=report_path())
    args = parser.parse_args(argv)
    report = write_report(args.duckdb, args.out, dbt_tolerance())
    print(
        f"PARITY {report['feature_version']}: {report['rows_joined']} rows joined "
        f"({report['rows_sql']} SQL, {report['rows_module']} module); "
        f"max |diff| doubles {report['max_abs_diff_doubles']!r}, integers "
        f"{report['max_abs_diff_integers']!r}, null mismatches {report['total_null_mismatches']}; "
        f"tolerance {report['tolerance']!r} -> {args.out}"
    )
    worst = max(
        (v["max_abs_diff"] or 0.0) for v in report["features"].values() if v["kind"] == "double"
    )
    if worst > report["tolerance"] or report["total_null_mismatches"]:
        print("PARITY: observed differences exceed the tolerance; the dbt test will fail")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
