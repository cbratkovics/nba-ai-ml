"""Repeated live runs of the golden dates to measure how often the agent passes.

A single brief at temperature 0 still varies between runs, so one pass or fail on a
golden date says little. This runs each golden date N times against the provider,
applies the same grounding check the nightly job uses, and records per-date and overall
pass rates in reports/agent_pass_rates.json. Nothing is written to brief/ or pushed.

Grounding pass: the brief came back with status "ok" and no finding was dropped.
Golden pass: a surviving finding names the date's largest-points-residual player.
An "agent_unavailable" brief fails both.

With --no-fallback the runner is single-model by construction: a 429 on the pinned
model is recorded as status "rate_limited" (both checks failed) instead of being
answered by the fallback model. rate_limited runs are counted separately in the
overall block, so a capped run reads as incomplete rather than as a lower pass rate.

Spreading across days: 25 briefs are about half of the free tier's daily token cap, and
the first full run was capped after 10 (2026-09-12). A run therefore measures a subset of
the golden dates (--dates, or --only-incomplete --max-dates N for the next N dates that
are missing or rate-limited in the committed report) and merges the result into the
committed report per date. Every date records the run that produced it; the overall
block is recomputed from every date's runs, and `complete` is true only when every
golden date has a full, unlimited set of runs. --readme rewrites the README row between
the `<!-- pass-rates -->` markers from the merged report, so the README never carries a
typed number.

Usage:
    python -m nba.agent.pass_rates [--runs 5] [--pause 30] [--no-fallback]
                                   [--dates 2026-01-14,2026-03-10 | --only-incomplete --max-dates 1]
                                   [--out reports/agent_pass_rates.json] [--no-merge]
                                   [--readme README.md]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import Counter
from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from nba import config
from nba.agent import evals, loop, tools

PASS_RATES_PATH = config.REPORTS_DIR / "agent_pass_rates.json"
STATUS_RATE_LIMITED = "rate_limited"
README_START = "<!-- pass-rates -->"
README_END = "<!-- /pass-rates -->"
# Provider error strings embed the account's organisation id; it is an identifier, not a
# secret, but it has no business in a committed report.
ORG_PATTERN = re.compile(r"org_[A-Za-z0-9]+")


def _is_rate_limit(error: str | None) -> bool:
    """True for the loop's record of a provider 429 (groq.RateLimitError)."""
    return bool(error) and error.startswith("RateLimitError")


def scrub_error(error: str | None) -> str | None:
    return None if error is None else ORG_PATTERN.sub("org_REDACTED", error)


def run_once(ctx: tools.ToolContext, d: date, golden: dict[str, Any], chat: Any) -> dict[str, Any]:
    brief, trace = loop.run_agent(ctx, d, chat)
    brief = evals.apply_grounding(brief)
    if brief["status"] == loop.STATUS_UNAVAILABLE and _is_rate_limit(brief.get("error")):
        brief["status"] = STATUS_RATE_LIMITED
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
        "error": scrub_error(brief.get("error")),
    }


def _rates(runs: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(runs)
    g = sum(1 for r in runs if r["grounding_pass"])
    k = sum(1 for r in runs if r["golden_pass"])
    limited = sum(1 for r in runs if r["status"] == STATUS_RATE_LIMITED)
    return {
        "runs": n,
        "grounding_pass": g,
        "grounding_rate": round(g / n, 3) if n else None,
        "golden_pass": k,
        "golden_rate": round(k / n, 3) if n else None,
        "rate_limited": limited,
        "complete": limited == 0,
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
    dates: list[str] | None = None,
    runner: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the golden dates (all, or `dates`) `runs` times each; return the pass-rate report."""
    per_date: dict[str, Any] = {}
    all_runs: list[dict[str, Any]] = []
    first = True
    chosen = sorted(golden) if dates is None else [d for d in sorted(golden) if d in set(dates)]
    for iso in chosen:
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
            "measured": {
                "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
                **(runner or {}),
            },
            "runs_detail": results,
        }
        all_runs.extend(results)
    report = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "mode": "live",
        "model_id": config.GROQ_MODEL,
        "fallback_model_id": config.GROQ_MODEL_FALLBACK,
        "runs_per_date": runs,
        "golden_dates": sorted(golden),
        "dates": per_date,
        "overall": _rates(all_runs),
    }
    return finalize(report)


def finalize(report: dict[str, Any]) -> dict[str, Any]:
    """Recompute the overall block and completeness from the per-date results."""
    golden_dates = report.get("golden_dates") or sorted(report["dates"])
    all_runs = [r for iso in sorted(report["dates"]) for r in report["dates"][iso]["runs_detail"]]
    overall = _rates(all_runs)
    complete_dates = [
        iso
        for iso in golden_dates
        if iso in report["dates"]
        and report["dates"][iso]["complete"]
        and report["dates"][iso]["runs"] >= report["runs_per_date"]
    ]
    overall["complete"] = len(complete_dates) == len(golden_dates) and bool(golden_dates)
    overall["dates_complete"] = len(complete_dates)
    overall["dates_total"] = len(golden_dates)
    overall["runs_expected"] = report["runs_per_date"] * len(golden_dates)
    overall["runs_completed"] = sum(1 for r in all_runs if r["status"] != STATUS_RATE_LIMITED)
    report["golden_dates"] = golden_dates
    report["overall"] = overall
    return report


def incomplete_dates(report: dict[str, Any] | None, golden: dict[str, Any]) -> list[str]:
    """Golden dates with no results yet, fewer runs than expected, or a rate-limited run."""
    if not report:
        return sorted(golden)
    out = []
    for iso in sorted(golden):
        d = report.get("dates", {}).get(iso)
        if d is None or not d.get("complete") or d.get("runs", 0) < report.get("runs_per_date", 0):
            out.append(iso)
    return out


def merge_reports(existing: dict[str, Any] | None, new: dict[str, Any]) -> dict[str, Any]:
    """Replace the measured dates of `existing` with those of `new`; recompute the rest."""
    if existing is None:
        return finalize(dict(new))
    if existing.get("runs_per_date") != new["runs_per_date"]:
        raise ValueError(
            f"runs per date differ: {existing.get('runs_per_date')} vs {new['runs_per_date']}"
        )
    merged = dict(existing)
    merged["dates"] = {**existing.get("dates", {}), **new["dates"]}
    merged["golden_dates"] = sorted(
        set(existing.get("golden_dates", [])) | set(new["golden_dates"])
    )
    merged["generated_at"] = new["generated_at"]
    merged["model_id"] = new["model_id"]
    merged["fallback_model_id"] = new["fallback_model_id"]
    return finalize(merged)


def readme_row(report: dict[str, Any]) -> str:
    """The README results-table cell for the pass-rate row, rendered from the report."""
    o = report["overall"]
    n_dates = o.get("dates_total", len(report["dates"]))
    per_date = report["runs_per_date"]
    done = o.get("runs_completed", o["runs"] - o["rate_limited"])
    expected = o.get("runs_expected", per_date * n_dates)
    if o["complete"]:
        return (
            f"grounding {o['grounding_pass']} of {o['runs']}, golden {o['golden_pass']} of "
            f"{o['runs']} ({n_dates} dates × {per_date} runs, `{report['model_id']}`)"
        )
    return (
        f"incomplete: {done} of {expected} briefs completed "
        f"({o.get('dates_complete', 0)} of {n_dates} dates; {o['rate_limited']} rate-limited); "
        f"of those, grounding {o['grounding_pass']} of {done}, golden {o['golden_pass']} of {done}"
    )


def update_readme(path: Path, report: dict[str, Any]) -> bool:
    """Rewrite the text between the pass-rate markers. Returns True when the file changed."""
    text = path.read_text()
    start, end = text.find(README_START), text.find(README_END)
    if start < 0 or end < 0 or end < start:
        raise ValueError(f"{path} has no {README_START} … {README_END} markers")
    new = text[: start + len(README_START)] + readme_row(report) + text[end:]
    if new == text:
        return False
    path.write_text(new)
    return True


def runner_info() -> dict[str, Any]:
    """Where a measurement ran, so a report never claims an Actions run that did not happen."""
    run_id = os.environ.get("GITHUB_RUN_ID")
    if run_id:
        return {
            "runner": "github_actions",
            "run_id": run_id,
            "workflow": os.environ.get("GITHUB_WORKFLOW"),
            "ref": os.environ.get("GITHUB_REF_NAME"),
        }
    return {"runner": "local"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--pause", type=float, default=30.0, help="seconds between briefs")
    parser.add_argument("--golden", type=Path, default=evals.GOLDEN_PATH)
    parser.add_argument("--out", type=Path, default=PASS_RATES_PATH)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help="never answer with the fallback model; record a 429 as rate_limited",
    )
    parser.add_argument("--dates", default=None, help="comma-separated golden dates to measure")
    parser.add_argument(
        "--only-incomplete",
        action="store_true",
        help="measure only dates missing or rate-limited in the existing report",
    )
    parser.add_argument("--max-dates", type=int, default=None, help="cap the dates per run")
    parser.add_argument(
        "--no-merge", action="store_true", help="overwrite --out instead of merging per date"
    )
    parser.add_argument("--readme", type=Path, default=None, help="README to update from --out")
    args = parser.parse_args(argv)
    golden = evals.load_golden(args.golden)
    if not golden:
        print(f"PASSRATE: no golden dates in {args.golden}")
        return 1
    existing = None
    if args.out.exists() and not args.no_merge:
        existing = json.loads(args.out.read_text())
    dates: list[str] | None = None
    if args.dates:
        dates = [d.strip() for d in args.dates.split(",") if d.strip()]
        unknown = sorted(set(dates) - set(golden))
        if unknown:
            print(f"PASSRATE: not golden dates: {unknown}")
            return 1
    elif args.only_incomplete:
        dates = incomplete_dates(existing, golden)
    if args.max_dates is not None and dates is not None:
        dates = dates[: args.max_dates]
    if dates is not None and not dates:
        print("PASSRATE: nothing to run; every golden date is complete")
        if args.readme and existing:
            update_readme(args.readme, existing)
        return 0

    def ctx_factory(d: date) -> tools.ToolContext:
        # Run date = the day after the brief date, as agent.yml does for replay dates.
        return tools.ToolContext(
            root=args.root,
            data_dir=args.root / config.DATA_DIR,
            run_date=date.fromordinal(d.toordinal() + 1),
        )

    def chat_factory() -> Any:
        if args.no_fallback:
            return loop.GroqChat(fallback_model_id=None)
        return loop.GroqChat()

    report = measure(
        golden,
        args.runs,
        ctx_factory,
        chat_factory,
        pause=args.pause,
        dates=dates,
        runner=runner_info(),
    )
    report["fallback_enabled"] = not args.no_fallback
    if args.no_fallback:
        report["fallback_model_id"] = None
    report = merge_reports(existing, report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    if args.readme:
        update_readme(args.readme, report)
    o = report["overall"]
    print(
        f"PASSRATE overall: grounding {o['grounding_pass']}/{o['runs']}, "
        f"golden {o['golden_pass']}/{o['runs']}, rate_limited {o['rate_limited']}, "
        f"dates complete {o['dates_complete']}/{o['dates_total']}"
        f"{'' if o['complete'] else ' (INCOMPLETE)'}; wrote {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
