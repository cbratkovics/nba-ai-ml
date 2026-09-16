"""Calibrate the drift rule on the replay season's game dates by season position (ADR-0017).

For every game date of the holdout season the check runs exactly as the nightly job would
(window of the previous DRIFT_WINDOW_DAYS days, position-aware reference), giving PSI per
feature per date. Every one of those dates is normal data: the model was evaluated on them,
so any date on which the rule fires is a false positive. The report gives, per position,
the distribution of PSI and the false-positive count at each candidate threshold under the
min_features rule, and chooses the smallest candidate with zero false positives on every
position. Committing reports/drift_calibration_<season>.json is what lets the nightly job
HOLD (before that it can only WARN).

Usage:
    python -m nba.drift.calibrate [--duckdb .duckdb/nba.duckdb] [--season 2025-26]
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nba import config
from nba.drift import check, reference
from nba.drift.check import POSITIONS
from nba.models.evaluate import git_sha


def calibration_path(season: str = config.HOLDOUT_SEASON) -> Path:
    return config.REPORTS_DIR / config.DRIFT_CALIBRATION_TEMPLATE.format(season=season)


def load_calibration(path: Path | None = None) -> dict[str, Any] | None:
    path = path or (config.REPO_ROOT / calibration_path())
    if not path.exists():
        return None
    cal = json.loads(path.read_text())
    cal["file"] = path.relative_to(config.REPO_ROOT).as_posix() if path.is_absolute() else str(path)
    return cal


def run_checks(
    feats: pd.DataFrame, ref: dict[str, Any], season: str, dates: list[date]
) -> list[dict[str, Any]]:
    return [check.check(feats, d, ref, season_game_dates=dates) for d in dates]


def summarise(
    checks: list[dict[str, Any]],
    candidates: tuple[float, ...] = config.DRIFT_CANDIDATE_THRESHOLDS,
    min_features: int = config.DRIFT_MIN_FEATURES,
) -> dict[str, Any]:
    per_date = []
    for c in checks:
        values = c["psi"]
        arr = np.array(list(values.values()), dtype="float64") if values else np.array([])
        per_date.append(
            {
                "date": c["date"],
                "position": c["position"],
                "reference_mode": c["reference_mode"],
                "n_rows": c["window"]["n_rows"],
                "max_psi": float(arr.max()) if arr.size else None,
                "median_psi": float(np.median(arr)) if arr.size else None,
                "max_feature": max(values, key=values.get) if values else None,
                "n_flagged": {str(t): int((arr >= t).sum()) if arr.size else 0 for t in candidates},
            }
        )
    positions: dict[str, Any] = {}
    for pos in POSITIONS:
        rows = [p for p in per_date if p["position"] == pos]
        scored = [p for p in rows if p["max_psi"] is not None]
        all_psi = np.array(
            [v for c in checks if c["position"] == pos for v in c["psi"].values()],
            dtype="float64",
        )
        feature_max: dict[str, float] = {}
        for c in checks:
            if c["position"] == pos:
                for f, v in c["psi"].items():
                    feature_max[f] = max(feature_max.get(f, 0.0), v)
        positions[pos] = {
            "n_dates": len(rows),
            "n_dates_scored": len(scored),
            "first_date": rows[0]["date"] if rows else None,
            "last_date": rows[-1]["date"] if rows else None,
            "psi": {
                "median": float(np.median(all_psi)) if all_psi.size else None,
                "p90": float(np.quantile(all_psi, 0.9)) if all_psi.size else None,
                "max": float(all_psi.max()) if all_psi.size else None,
            },
            "top_features": sorted(feature_max.items(), key=lambda kv: -kv[1])[:5],
            "false_positives": {
                str(t): {
                    "count": sum(1 for p in scored if p["n_flagged"][str(t)] >= min_features),
                    "dates": [p["date"] for p in scored if p["n_flagged"][str(t)] >= min_features],
                }
                for t in candidates
            },
        }
    totals = {
        str(t): sum(positions[p]["false_positives"][str(t)]["count"] for p in positions)
        for t in candidates
    }
    zero = [t for t in candidates if totals[str(t)] == 0]
    chosen = min(zero) if zero else max(candidates)
    return {
        "candidates": list(candidates),
        "min_features": min_features,
        "false_positives_total": totals,
        "chosen": {
            "threshold": float(chosen),
            "min_features": min_features,
            "false_positives": totals[str(chosen)],
            "rule": (
                f"HOLD when at least {min_features} features have PSI >= {chosen}; the smallest "
                "candidate with zero false positives on every season position"
                if zero
                else f"no candidate reached zero false positives; largest candidate {chosen}"
            ),
        },
        "positions": positions,
        "per_date": per_date,
    }


def build_calibration(
    feats: pd.DataFrame,
    ref: dict[str, Any],
    season: str,
    *,
    generated_at: str,
    git_sha: str,
) -> dict[str, Any]:
    in_season = feats[feats["season"] == season]
    dates = sorted({t.date() for t in in_season["game_date"].unique()})
    checks = run_checks(feats, ref, season, dates)
    summary = summarise(checks)
    return {
        "kind": "drift_calibration",
        "season": season,
        "generated_at": generated_at,
        "git_sha": git_sha,
        "feature_version": ref["feature_version"],
        "model_revision": ref["model_revision"],
        "reference_file": reference.reference_path().as_posix(),
        "window_days": config.DRIFT_WINDOW_DAYS,
        "min_rows": config.DRIFT_MIN_ROWS,
        "n_dates": len(dates),
        "note": (
            "Every date is normal data (the replay season the model was evaluated on), so a "
            "date on which the rule fires is a false positive. Regular season only; the data "
            "has no playoffs. Opening windows use the opening reference (ADR-0017)."
        ),
        **summary,
    }


def validate_calibration(cal: dict[str, Any]) -> list[str]:
    problems = []
    for key in ("season", "chosen", "positions", "per_date", "candidates", "min_features"):
        if key not in cal:
            problems.append(f"missing key {key}")
    if not problems:
        if cal["chosen"]["threshold"] not in cal["candidates"]:
            problems.append("chosen threshold not among the candidates")
        if len(cal["per_date"]) != cal.get("n_dates"):
            problems.append("per_date length differs from n_dates")
        for pos in POSITIONS:
            if pos not in cal["positions"]:
                problems.append(f"missing position {pos}")
    return problems


def write_calibration(
    duckdb_path: Path = reference.DEFAULT_DUCKDB,
    season: str = config.HOLDOUT_SEASON,
    out: Path | None = None,
) -> dict[str, Any]:
    ref = reference.load_reference()
    if ref is None:
        raise SystemExit("DRIFT: no reference; run python -m nba.drift.reference first")
    rows = reference.load_gold_rows(duckdb_path)
    logs = rows.drop(columns=["population", "dataset_revision"])
    feats = check.population_rows(logs)
    cal = build_calibration(
        feats,
        ref,
        season,
        generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
        git_sha=git_sha(),
    )
    problems = validate_calibration(cal)
    if problems:
        raise SystemExit("DRIFT: calibration failed validation: " + "; ".join(problems))
    out = out or calibration_path(season)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cal, indent=2) + "\n")
    return cal


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--duckdb", type=Path, default=reference.DEFAULT_DUCKDB)
    parser.add_argument("--season", default=config.HOLDOUT_SEASON)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    cal = write_calibration(args.duckdb, args.season, args.out)
    print(f"DRIFT calibration {args.season}: wrote {args.out or calibration_path(args.season)}")
    print(f"DRIFT chosen: {cal['chosen']}")
    for pos, block in cal["positions"].items():
        fps = {t: v["count"] for t, v in block["false_positives"].items()}
        print(
            f"DRIFT {pos}: dates={block['n_dates']} scored={block['n_dates_scored']} "
            f"psi median={block['psi']['median']} p90={block['psi']['p90']} "
            f"max={block['psi']['max']} fps={fps} top={block['top_features'][:3]}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
