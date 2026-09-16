"""Write the policy evaluation artifact, reports/policy_<season>.json, from the gold marts.

Reads gold.fct_prediction joined to gold.fct_player_game (the season-to-date mean the
second baseline needs) from a built local warehouse, keeps the holdout season's replay
rows, and evaluates the policy per population (nba/decisions/policy.py). Also derives
frontend/lib/policy_summary.json, the file the /decisions page imports; a test recomputes
it from the artifact and compares, like the all-rows baseline.

Sequence (the dbt policy marts read the artifact through bronze, so it is written between
two builds):

    make dbt-full          # fct_prediction and fct_player_game exist; the policy marts are empty
    make policy            # this module: writes the artifact and the site summary
    make dbt-full          # the policy marts apply the thresholds and reconcile to the artifact

Usage:
    python -m nba.decisions.evaluate [--duckdb .duckdb/nba.duckdb] [--season 2025-26]
                                     [--out reports/policy_<season>.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from nba import config
from nba.decisions import policy
from nba.models.evaluate import git_sha
from nba.warehouse import load

SITE_SUMMARY_PATH = Path("frontend") / "lib" / "policy_summary.json"
DEFAULT_DUCKDB = Path(".duckdb") / "nba.duckdb"

ROWS_SQL = """
select
    p.player_id, p.game_id, p.model_revision, p.feature_version, p.dataset_revision,
    p.run_kind, p.season, p.game_date, p.has_actual, p.baseline_defined,
    p.in_metrics_population,
    p.pred_pts, p.pred_reb, p.pred_ast,
    p.baseline_last10_pts, p.baseline_last10_reb, p.baseline_last10_ast,
    p.actual_pts, p.actual_reb, p.actual_ast,
    g.pts_mean_season_prior as baseline_season_pts,
    g.reb_mean_season_prior as baseline_season_reb,
    g.ast_mean_season_prior as baseline_season_ast
from gold.fct_prediction as p
left join gold.fct_player_game as g on p.player_id = g.player_id and p.game_id = g.game_id
where p.run_kind = 'replay' and p.season = ?
order by p.game_date, p.game_id, p.player_id
"""


def report_path(season: str = config.HOLDOUT_SEASON) -> Path:
    return config.REPORTS_DIR / config.POLICY_REPORT_TEMPLATE.format(season=season)


def load_replay_rows(duckdb_path: Path, season: str) -> pd.DataFrame:
    import duckdb

    con = duckdb.connect(str(duckdb_path), read_only=True)
    try:
        return con.execute(ROWS_SQL, [season]).df()
    finally:
        con.close()


def input_description(duckdb_path: Path, rows: pd.DataFrame, warehouse_root: Path) -> dict:
    manifest_path = warehouse_root / load.MANIFEST_FILE
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    return {
        "source": "gold.fct_prediction left join gold.fct_player_game on (player_id, game_id), "
        "run_kind = 'replay'",
        "duckdb": str(duckdb_path),
        "n_rows": int(len(rows)),
        "n_with_actual": int(rows["has_actual"].sum()),
        "n_min10": int(rows["in_metrics_population"].sum()),
        "warehouse_load_manifest": {
            k: manifest[k]
            for k in ("loaded_at", "dataset_repo", "dataset_revision", "families")
            if manifest and k in manifest
        }
        if manifest
        else None,
    }


def write_artifact(
    duckdb_path: Path = DEFAULT_DUCKDB,
    season: str = config.HOLDOUT_SEASON,
    out: Path | None = None,
    warehouse_root: Path = load.DEFAULT_ROOT,
) -> dict[str, Any]:
    rows = load_replay_rows(duckdb_path, season)
    if rows.empty:
        raise SystemExit(f"POLICY: no replay rows for {season} in {duckdb_path}; run dbt build")
    artifact = policy.build_artifact(
        rows,
        season,
        generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
        git_sha=git_sha(),
        input_description=input_description(duckdb_path, rows, warehouse_root),
    )
    problems = policy.validate_artifact(artifact)
    if problems:
        raise SystemExit("POLICY: artifact failed validation: " + "; ".join(problems))
    out = out or report_path(season)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2) + "\n")
    return artifact


def site_summary_from_report(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    artifact = json.loads(raw)
    rel = path.resolve().relative_to(config.REPO_ROOT).as_posix()
    return policy.site_summary(artifact, rel, hashlib.sha256(raw).hexdigest())


def write_site_summary(path: Path, out: Path = SITE_SUMMARY_PATH) -> dict[str, Any]:
    summary = site_summary_from_report(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--duckdb", type=Path, default=DEFAULT_DUCKDB)
    parser.add_argument("--season", default=config.HOLDOUT_SEASON)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--warehouse-root", type=Path, default=load.DEFAULT_ROOT)
    args = parser.parse_args(argv)
    out = args.out or report_path(args.season)
    artifact = write_artifact(args.duckdb, args.season, out, args.warehouse_root)
    print(f"POLICY {args.season}: wrote {out}")
    for p, block in artifact["populations"].items():
        for t, tb in block["targets"].items():
            print(
                f"POLICY {p} {t}: n={tb['n']} threshold={tb['threshold']} "
                f"coverage={tb['coverage']:.3f} hit={tb['hit_rate']:.4f} "
                f"season_mean_same_rows={tb['baselines']['season_mean_sign']['hit_rate']:.4f} "
                f"beats_both={tb['model_beats_both']}"
            )
    if out == report_path(args.season):
        write_site_summary(out)
        print(f"POLICY {args.season}: wrote {SITE_SUMMARY_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
