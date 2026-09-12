"""Repeated live runs of the golden dates to measure how often the agent passes.

A single brief at temperature 0 still varies between runs, so one pass or fail on a
golden date says little. This runs each golden date N times against the provider,
applies the same grounding check the nightly job uses, and records per-date and overall
pass rates in reports/agent_pass_rates.json. Nothing is written to brief/ or pushed.

Grounding pass: the brief came back with status "ok" and no finding was dropped.
Golden pass: a surviving finding names the date's largest-points-residual player.
An "agent_unavailable" brief fails both.

Usage:
    python -m nba.agent.pass_rates [--runs 5] [--pause 30] [--out reports/agent_pass_rates.json]
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from nba import config
from nba.agent import evals, loop, tools

PASS_RATES_PATH = config.REPORTS_DIR / "agent_pass_rates.json"


def run_once(ctx: tools.ToolContext, d: date, golden: dict[str, Any], chat: Any) -> dict[str, Any]:
    brief, trace = loop.run_agent(ctx, d, chat)
    brief = evals.apply_grounding(brief)
    grounding_pass = brief["status"] == loop.STATUS_OK
    golden_pass = bool(brief["findings"]) and evals.golden_hit(
        brief, golden["player_name"], golden.get("player_id")
    )
    return {
        "status": brief["status"],
        "grounding_pass": grounding_pass,
        "golden_pass": golden_pass,
        "n_findings": len(brief["findings"]),
        "n_dropped": len(brief.get("dropped_findings", [])),
        "model_id": brief.get("model_id"),
        "tool_calls_made": brief.get("tool_calls_made"),
        "latency_ms": brief.get("latency_ms"),
        "error": brief.get("error"),
    }


def _rates(runs: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(runs)
    g = sum(1 for r in runs if r["grounding_pass"])
    k = sum(1 for r in runs if r["golden_pass"])
    return {
        "runs": n,
        "grounding_pass": g,
        "grounding_rate": round(g / n, 3) if n else None,
        "golden_pass": k,
        "golden_rate": round(k / n, 3) if n else None,
        "statuses": dict(Counter(r["status"] for r in runs)),
        "models": dict(Counter(r["model_id"] for r in runs)),
    }


def measure(
    golden: dict[str, dict[str, Any]],
    runs: int,
    ctx_factory: Callable[[date], tools.ToolContext],
    chat_factory: Callable[[], Any],
    pause: float = 0.0,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Run every golden date `runs` times; return the pass-rate report."""
    per_date: dict[str, Any] = {}
    all_runs: list[dict[str, Any]] = []
    first = True
    for iso in sorted(golden):
        d = date.fromisoformat(iso)
        results = []
        for i in range(runs):
            if not first and pause:
                time.sleep(pause)
            first = False
            r = run_once(ctx_factory(d), d, golden[iso], chat_factory())
            r["run"] = i + 1
            results.append(r)
            log(
                f"PASSRATE {iso} run {i + 1}/{runs}: status={r['status']} "
                f"grounding={'pass' if r['grounding_pass'] else 'FAIL'} "
                f"golden={'pass' if r['golden_pass'] else 'FAIL'} model={r['model_id']} "
                f"latency_ms={r['latency_ms']}"
            )
        per_date[iso] = {
            "player_name": golden[iso]["player_name"],
            "player_id": golden[iso].get("player_id"),
            **_rates(results),
            "runs_detail": results,
        }
        all_runs.extend(results)
    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "mode": "live",
        "model_id": config.GROQ_MODEL,
        "fallback_model_id": config.GROQ_MODEL_FALLBACK,
        "runs_per_date": runs,
        "dates": per_date,
        "overall": _rates(all_runs),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--pause", type=float, default=30.0, help="seconds between briefs")
    parser.add_argument("--golden", type=Path, default=evals.GOLDEN_PATH)
    parser.add_argument("--out", type=Path, default=PASS_RATES_PATH)
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    golden = evals.load_golden(args.golden)
    if not golden:
        print(f"PASSRATE: no golden dates in {args.golden}")
        return 1

    def ctx_factory(d: date) -> tools.ToolContext:
        # Run date = the day after the brief date, as agent.yml does for replay dates.
        return tools.ToolContext(
            root=args.root,
            data_dir=args.root / config.DATA_DIR,
            run_date=date.fromordinal(d.toordinal() + 1),
        )

    report = measure(golden, args.runs, ctx_factory, loop.GroqChat, pause=args.pause)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    o = report["overall"]
    print(
        f"PASSRATE overall: grounding {o['grounding_pass']}/{o['runs']}, "
        f"golden {o['golden_pass']}/{o['runs']}; wrote {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
