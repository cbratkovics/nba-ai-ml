"""Write and render the holdout metrics report (reports/metrics.json).

Also derives the all-rows baseline summary the site shows: the row-weighted season MAE of
the model and of the last-10 baseline over every replayed day, computed from the committed
copy of the replay's daily file (reports/replay_all_rows_<season>.json) and written to
frontend/lib/all_rows_baseline.json so the pages import a committed number instead of
computing one from a fetch. A test recomputes it from the report and compares.

Usage:
    python -m nba.models.evaluate all-rows [--season 2025-26]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from nba import config

METRIC_KEYS = ("mae", "rmse", "r2", "n")
ALL_ROWS_REPORT_TEMPLATE = "replay_all_rows_{season}.json"
ALL_ROWS_SUMMARY_PATH = Path("frontend") / "lib" / "all_rows_baseline.json"
ALL_ROWS_DECIMAL_PLACES = 12


def _weighted_mean(days: list[dict[str, Any]], key: str, target: str, n: int) -> float | None:
    """Return a reproducible weighted mean from the report's decimal JSON values.

    Summing binary floats can differ by a few units in the last place between Python
    versions.  The site does not need more than twelve decimal places, so aggregate the
    source decimals exactly and round once at that documented serialization boundary.
    """
    if not n:
        return None
    value = sum((d[key][target] * int(d["n"]) for d in days), start=Decimal(0)) / n
    quantum = Decimal(1).scaleb(-ALL_ROWS_DECIMAL_PLACES)
    return float(value.quantize(quantum))


def all_rows_report_path(season: str = config.HOLDOUT_SEASON) -> Path:
    return config.REPORTS_DIR / ALL_ROWS_REPORT_TEMPLATE.format(season=season)


def all_rows_summary(report_path: Path) -> dict[str, Any]:
    """Row-weighted season MAE per target for the model and the last-10 baseline."""
    raw = report_path.read_bytes()
    daily = json.loads(raw, parse_float=Decimal)
    days = daily["days"]
    n = sum(int(d["n"]) for d in days)
    targets = list(daily.get("targets", config.TARGETS))
    model = {t: _weighted_mean(days, "model", t, n) for t in targets}
    baseline = {t: _weighted_mean(days, "baseline_last10", t, n) for t in targets}
    try:
        source_file = report_path.resolve().relative_to(config.REPO_ROOT).as_posix()
    except ValueError:
        source_file = report_path.as_posix()
    return {
        "season": daily["season"],
        "population": daily["population"],
        "source_file": source_file,
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "n_dates": int(daily["n_dates"]),
        "first_date": daily.get("first_date"),
        "last_date": daily.get("last_date"),
        "n": n,
        "model_mae": model,
        "baseline_last10_mae": baseline,
        "baseline_wins": [t for t in targets if baseline[t] is not None and baseline[t] < model[t]],
    }


def write_all_rows_summary(report_path: Path, out: Path = ALL_ROWS_SUMMARY_PATH) -> dict[str, Any]:
    summary = all_rows_summary(report_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def build_payload(results: dict[str, Any], dataset_version: str) -> dict[str, Any]:
    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_sha": git_sha(),
        "dataset": {"repo": config.HF_DATASET_REPO, "version": dataset_version},
        "split": results["split"],
        "features": results["features"],
        "lgbm_params": results["lgbm_params"],
        "model_files": results["model_files"],
        "metrics": results["targets"],
    }


def write_metrics(results: dict[str, Any], path: Path, dataset_version: str) -> dict[str, Any]:
    payload = build_payload(results, dataset_version)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def read_metrics(path: Path = config.METRICS_PATH) -> dict[str, Any]:
    return json.loads(path.read_text())


def metrics_table(payload: dict[str, Any]) -> str:
    """Markdown table of model vs baselines per target on the holdout season."""
    split = payload["split"]
    dates = split["holdout_dates"]
    lines = [
        f"Holdout season: {split['holdout_season']} ({dates['start']} to {dates['end']})",
        "",
        "| Target | Predictor | MAE | RMSE | R² | n |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for target, predictors in payload["metrics"].items():
        for name, m in predictors.items():
            lines.append(
                f"| {target} | {name} | {m['mae']:.3f} | {m['rmse']:.3f} | "
                f"{m['r2']:.3f} | {m['n']} |"
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p_all = sub.add_parser("all-rows", help="derive the all-rows baseline summary for the site")
    p_all.add_argument("--season", default=config.HOLDOUT_SEASON)
    p_all.add_argument("--out", type=Path, default=ALL_ROWS_SUMMARY_PATH)
    args = parser.parse_args(argv)
    if args.command == "all-rows":
        summary = write_all_rows_summary(all_rows_report_path(args.season), args.out)
        for t, m in summary["model_mae"].items():
            print(f"ALLROWS {t}: model {m:.4f} last10 {summary['baseline_last10_mae'][t]:.4f}")
        print(f"ALLROWS n={summary['n']} wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
