"""What the workflow should open an issue about, from data/nightly_summary.json.

Prints nothing when the run needs no issue; otherwise one line per issue:

    NOTIFY <label> <title>

and writes the body to --body-out. The workflow creates the label idempotently
(`gh label create <label> --force`) and skips the issue when an open one has the same
title, so a HOLD that persists for days opens one issue, not one per night.

Usage:
    python -m nba.drift.notify [--summary data/nightly_summary.json] [--body-out issue_body.md]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SUMMARY_PATH = Path("data") / "nightly_summary.json"
HOLD_LABEL = "nightly-hold"
WARN_LABEL = "nightly-warn"


def issues(summary: dict[str, Any]) -> list[dict[str, str]]:
    drift = summary.get("drift") or {}
    out: list[dict[str, str]] = []
    status = drift.get("status")
    date = summary.get("date", "?")
    reasons = "\n".join(f"- {r}" for r in drift.get("reasons", [])) or "- (no reasons recorded)"
    if status == "hold":
        out.append(
            {
                "label": HOLD_LABEL,
                "title": "Nightly HOLD: feature drift",
                "body": (
                    f"Run date {date}: the calibrated drift rule fired "
                    f"({drift.get('n_flagged')} features at or above PSI "
                    f"{(drift.get('thresholds') or {}).get('psi')}).\n\n{reasons}\n\n"
                    "The slate and decisions were still written (HOLD does not block the "
                    "slate). Inspect drift/<date>.json in the dataset repo and mart_drift."
                ),
            }
        )
    if drift.get("no_schedule_warn"):
        out.append(
            {
                "label": WARN_LABEL,
                "title": "Nightly WARN: no schedule file for the season",
                "body": (
                    f"Run date {date}: {drift.get('no_schedule_streak')} consecutive nightly "
                    "runs ended at the no-schedule line. The Kaggle dump may have stopped "
                    "publishing the season's LeagueSchedule file (AUDIT.md risk 5).\n\n"
                    "Check the dump's file list; until a schedule exists no slate can be "
                    "written."
                ),
            }
        )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--summary", type=Path, default=SUMMARY_PATH)
    parser.add_argument("--body-out", type=Path, default=Path("issue_body.md"))
    args = parser.parse_args(argv)
    if not args.summary.exists():
        return 0
    summary = json.loads(args.summary.read_text())
    found = issues(summary)
    if not found:
        return 0
    args.body_out.write_text("\n\n---\n\n".join(i["body"] for i in found) + "\n")
    for i in found:
        print(f"NOTIFY {i['label']} {i['title']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
