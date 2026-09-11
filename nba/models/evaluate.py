"""Write and render the holdout metrics report (reports/metrics.json)."""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from nba import config

METRIC_KEYS = ("mae", "rmse", "r2", "n")


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
