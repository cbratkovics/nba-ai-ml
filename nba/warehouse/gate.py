"""Decide whether tonight's warehouse build has anything to do.

The nightly job writes data/nightly_summary.json. In the off-season every run ingests zero
new or changed rows, scores no residuals and writes no slate; building the warehouse then
would only re-copy yesterday's tables (fixed decision 7: zero-row runs exit 0 without a
build). Prints one line and exits 0 either way; the workflow reads the first word.

    WAREHOUSE build: <reason>
    WAREHOUSE skip: <reason>

Usage:
    python -m nba.warehouse.gate [--summary data/nightly_summary.json] [--force]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SUMMARY_PATH = Path("data") / "nightly_summary.json"


def decide(summary: dict[str, Any], force: bool = False) -> tuple[bool, str]:
    if force:
        return True, "forced"
    counts = summary.get("ingest", {}).get("counts", {}) or {}
    changed = int(counts.get("new", 0) or 0) + int(counts.get("changed", 0) or 0)
    residuals = summary.get("residuals", {}) or {}
    scored = "n_with_actuals" in residuals
    slate_ok = (summary.get("slate", {}) or {}).get("status") == "ok"
    reasons = []
    if changed:
        reasons.append(f"{changed} new or changed game-log rows")
    if scored:
        reasons.append("residuals scored")
    if slate_ok:
        reasons.append("slate written")
    if reasons:
        return True, ", ".join(reasons)
    return False, "no new rows, no residuals, no slate (off-season zero-row run)"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--summary", type=Path, default=SUMMARY_PATH)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if not args.summary.exists():
        print(f"WAREHOUSE build: no summary at {args.summary}")
        return 0
    build, reason = decide(json.loads(args.summary.read_text()), force=args.force)
    print(f"WAREHOUSE {'build' if build else 'skip'}: {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
