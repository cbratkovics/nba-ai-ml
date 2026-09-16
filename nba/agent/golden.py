"""Rebuild reports/agent_golden.json (version 2) from the products the tools read.

Each golden date keeps its largest-points-residual player (version 1) and gains a decision
fact and a drift fact computed by the same tool functions the agent calls, with the run
date one day after the brief date as agent.yml and the pass-rate runner use:

    decision_fact  get_rolling_metrics(...)["decisions"]["pts"] hit_rate and n_resolved to
                   date on the training population (fct_decision_policy when exported, else
                   the residual files with the committed policy artifact)
    drift_fact     get_daily_report(run_date)["drift"] status word on or before the run date
                   (mart_drift, else drift/<date>.json, else the calibration's per-date row
                   for replay dates); off-season dates carry the insufficient / streak line

Usage:
    python -m nba.agent.golden [--root .] [--data-dir data/game_logs]
                               [--golden reports/agent_golden.json]
"""

from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from nba import config
from nba.agent import evals, tools

GOLDEN_VERSION = 2


def facts_for(ctx: tools.ToolContext, d: date) -> tuple[dict[str, Any], dict[str, Any]]:
    rolling = tools.get_rolling_metrics(ctx, tools.ROLLING_DEFAULT_DAYS)
    dec = rolling.get("decisions", {})
    pts = dec.get("pts") if dec.get("available") else None
    decision_fact = {
        "tool": "get_rolling_metrics",
        "path": "decisions.pts.hit_rate",
        "value": pts["hit_rate"] if pts else None,
        "n_resolved": pts["n_resolved"] if pts else None,
        "through": dec.get("through"),
        "population": "min10",
        "source": dec.get("source"),
    }
    report = tools.get_daily_report(ctx, ctx.run_date.isoformat())
    drift = report["drift"]
    drift_fact = {
        "tool": "get_daily_report",
        "path": "drift.status",
        "status": drift.get("status") if drift.get("available") else "no evidence",
        "as_of": drift.get("as_of"),
        "flagged": drift.get("flagged", []),
        "no_schedule_streak": drift.get("no_schedule_streak"),
        "source": drift.get("source"),
    }
    return decision_fact, drift_fact


def rebuild(golden_path: Path, root: Path, data_dir: Path) -> dict[str, Any]:
    golden = json.loads(golden_path.read_text())
    for entry in golden["dates"]:
        d = date.fromisoformat(entry["date"])
        ctx = tools.ToolContext(root=root, data_dir=data_dir, run_date=d + timedelta(days=1))
        decision_fact, drift_fact = facts_for(ctx, d)
        entry["decision_fact"] = decision_fact
        entry["drift_fact"] = drift_fact
    golden["version"] = GOLDEN_VERSION
    golden["facts"] = {
        "player": "a finding names the player with the largest absolute points residual",
        "decision": "a finding cites get_rolling_metrics and carries decisions.pts.hit_rate to "
        "date (within 0.01, or as a percentage within 0.5)",
        "drift": "a finding cites get_daily_report and names the drift status word",
        "run_date": "the day after the brief date",
    }
    return golden


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    parser.add_argument("--golden", type=Path, default=evals.GOLDEN_PATH)
    args = parser.parse_args(argv)
    golden = rebuild(args.golden, args.root, args.data_dir)
    args.golden.write_text(json.dumps(golden, indent=2) + "\n")
    for e in golden["dates"]:
        print(
            f"GOLDEN {e['date']}: {e['player_name']}; decision pts hit_rate="
            f"{e['decision_fact']['value']} (n={e['decision_fact']['n_resolved']}, "
            f"{e['decision_fact']['source']}); drift={e['drift_fact']['status']} "
            f"as_of={e['drift_fact']['as_of']} ({e['drift_fact']['source']})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
