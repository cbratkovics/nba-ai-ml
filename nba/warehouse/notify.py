"""What the workflow should open an issue about after a dbt build, from run_results.json.

Severity policy (ADR-0026): a test that would publish a wrong number is `error` and fails the
build; an operational threshold (restatement lag beyond the lookback, a season missing from
the seed, an implausible box score) is `warn`, never blocks, and lands here. Prints nothing
when no test warned; otherwise one line per warning test, in the protocol nba/drift/notify.py
uses so the workflow step is the same loop:

    NOTIFY <label> <title>

and writes the body to --body-out. The title is stable per test, so a warning that persists
across nights reuses its open issue instead of opening one per night.

Usage:
    python -m nba.warehouse.notify [--run-results dbt/target/run_results.json]
                                   [--body-out issue_body.md]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

RUN_RESULTS_PATH = Path("dbt") / "target" / "run_results.json"
WARN_LABEL = "warehouse-warn"
RUNBOOK = "docs/RUNBOOK.md"


def warnings(run_results: dict[str, Any]) -> list[dict[str, Any]]:
    """The test results whose status is warn, oldest first."""
    out = []
    for r in run_results.get("results", []):
        if r.get("status") != "warn":
            continue
        name = r["unique_id"].split(".")[2] if r["unique_id"].count(".") >= 2 else r["unique_id"]
        out.append(
            {
                "name": name,
                "failures": r.get("failures"),
                "message": (r.get("message") or "").strip(),
            }
        )
    return out


def issues(run_results: dict[str, Any]) -> list[dict[str, str]]:
    generated = (run_results.get("metadata") or {}).get("generated_at", "?")
    target = (run_results.get("args") or {}).get("target", "?")
    out: list[dict[str, str]] = []
    for w in warnings(run_results):
        out.append(
            {
                "label": WARN_LABEL,
                "title": f"Warehouse WARN: {w['name']}",
                "body": (
                    f"dbt build at {generated} (target `{target}`): test `{w['name']}` "
                    f"returned {w['failures']} row(s) with severity warn.\n\n"
                    f"{w['message'] or '(no message)'}\n\n"
                    f"What this means and what to do: {RUNBOOK} (section on `{w['name']}`). "
                    "The build and the export were not blocked (ADR-0026)."
                ),
            }
        )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-results", type=Path, default=RUN_RESULTS_PATH)
    parser.add_argument("--body-out", type=Path, default=Path("issue_body.md"))
    args = parser.parse_args(argv)
    if not args.run_results.exists():
        return 0
    found = issues(json.loads(args.run_results.read_text()))
    if not found:
        return 0
    args.body_out.write_text("\n\n---\n\n".join(i["body"] for i in found) + "\n")
    for i in found:
        print(f"NOTIFY {i['label']} {i['title']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
