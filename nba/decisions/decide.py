"""Apply the committed policy to a slate: decisions/<date>.json and decisions/latest.json.

Every slate row gets, per target, the edge (prediction - last-10 mean), the call under each
population's policy and the bands, read from reports/policy_<season>.json (the artifact the
warehouse reconciles to). Nothing here needs a box score; the warehouse mart
fct_decision_policy resolves the calls once the residuals exist, using the same thresholds
from the same file. The nightly job writes this right after the slate; when the artifact is
missing it logs one line and writes nothing (the slate itself is unaffected).

Usage:
    python -m nba.decisions.decide --date YYYY-MM-DD [--predictions-dir predictions]
                                   [--decisions-dir decisions]
                                   [--report reports/policy_<season>.json]
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from nba import config
from nba.decisions import evaluate, policy

LATEST_FILE = "latest.json"


def load_artifact(path: Path | None = None) -> dict[str, Any] | None:
    path = path or (config.REPO_ROOT / evaluate.report_path())
    return json.loads(path.read_text()) if path.exists() else None


def decide_slate(predictions: pd.DataFrame, d: date, artifact: dict[str, Any]) -> dict[str, Any]:
    applied = {
        (p, t): policy.apply_policy(predictions, artifact, p, t)
        for p in policy.POPULATIONS
        for t in config.TARGETS
    }
    rows = []
    for i, r in enumerate(predictions.itertuples(index=False)):
        targets: dict[str, Any] = {}
        for t in config.TARGETS:
            per_pop = {}
            for p in policy.POPULATIONS:
                a = applied[(p, t)].iloc[i]
                per_pop[p] = {
                    "call": str(a["call"]),
                    "band_50": [round(float(a["low_50"]), 2), round(float(a["high_50"]), 2)],
                    "band_80": [round(float(a["low_80"]), 2), round(float(a["high_80"]), 2)],
                }
            edge = applied[("min10", t)].iloc[i]["edge"]
            targets[t] = {
                "prediction": round(float(getattr(r, f"pred_{t}")), 2),
                "baseline_last10": round(float(getattr(r, f"{t}_mean_last10")), 2),
                "edge": None if pd.isna(edge) else round(float(edge), 2),
                "populations": per_pop,
            }
        rows.append(
            {
                "game_id": str(r.game_id),
                "player_id": int(r.player_id),
                "player_name": str(r.player_name),
                "team": str(r.team),
                "opponent": str(r.opponent),
                "home": bool(r.home),
                "targets": targets,
            }
        )
    policies = {
        p: {
            "description": block["description"],
            "targets": {
                t: {
                    "threshold": tb["threshold"],
                    "hit_rate_in_sample": tb["hit_rate"],
                    "coverage_in_sample": tb["coverage"],
                    "model_beats_both": tb["model_beats_both"],
                    "verdict": tb["verdict"],
                    "band_quantiles": tb["bands"]["quantiles"],
                }
                for t, tb in block["targets"].items()
            },
        }
        for p, block in artifact["populations"].items()
    }
    n_calls = {
        p: {t: int((applied[(p, t)]["call"] != "no_call").sum()) for t in config.TARGETS}
        for p in policy.POPULATIONS
    }
    return {
        "date": d.isoformat(),
        "model_revision": str(predictions["model_revision"].iloc[0]),
        "dataset_revision": str(predictions["dataset_revision"].iloc[0]),
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "policy": {
            "season": artifact["season"],
            "report": evaluate.report_path(artifact["season"]).as_posix(),
            "git_sha": artifact["git_sha"],
            "in_sample": artifact["in_sample"],
            "populations": policies,
        },
        "n_games": int(predictions["game_id"].nunique()),
        "n_players": int(len(predictions)),
        "n_calls": n_calls,
        "rows": rows,
    }


def write_outputs(
    predictions: pd.DataFrame, d: date, decisions_dir: Path, artifact: dict[str, Any]
) -> list[Path]:
    """decisions/<date>.json and decisions/latest.json (the same content)."""
    decisions_dir.mkdir(parents=True, exist_ok=True)
    payload = decide_slate(predictions, d, artifact)
    text = json.dumps(payload, indent=2) + "\n"
    dated = decisions_dir / f"{d.isoformat()}.json"
    latest = decisions_dir / LATEST_FILE
    dated.write_text(text)
    latest.write_text(text)
    return [dated, latest]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--date", type=date.fromisoformat, required=True)
    parser.add_argument("--predictions-dir", type=Path, default=config.PREDICTIONS_DIR)
    parser.add_argument("--decisions-dir", type=Path, default=config.DECISIONS_DIR)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args(argv)
    artifact = load_artifact(args.report)
    if artifact is None:
        print(f"DECISIONS {args.date}: no policy report; nothing written")
        return 0
    pred_path = args.predictions_dir / f"{args.date.isoformat()}.parquet"
    if not pred_path.exists():
        print(f"DECISIONS {args.date}: no predictions file {pred_path}; nothing written")
        return 0
    written = write_outputs(pd.read_parquet(pred_path), args.date, args.decisions_dir, artifact)
    print(f"DECISIONS {args.date}: wrote {[str(p) for p in written]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
