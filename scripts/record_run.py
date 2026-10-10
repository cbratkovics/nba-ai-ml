#!/usr/bin/env python
"""Record a GitHub Actions run in reports/runs.json, the run-evidence artifact (ADR-0022).

Every "ran" claim in the README, the ADRs and the workflow headers must name a run id, and
tests/test_runs.py fails when a cited id is missing from this file. Fetches the run's
metadata with `gh api` so nothing is typed by hand; --note, --target and --pushed add what
the metadata cannot know (which warehouse target the run chose, which Hugging Face revision
it pushed).

    python scripts/record_run.py 37201223219 --target motherduck
    python scripts/record_run.py 35048092357 --target motherduck \\
        --pushed cbratkovics/nba-game-logs@66404be0 --note "first gold/ export"
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

RUNS_PATH = Path("reports") / "runs.json"
REPO = "cbratkovics/nba-ai-ml"
FIELDS = ("run_id", "workflow", "event", "branch", "commit", "conclusion", "started_at")


def fetch(run_id: int, repo: str = REPO) -> dict:
    raw = subprocess.run(
        ["gh", "api", f"repos/{repo}/actions/runs/{run_id}"], capture_output=True, text=True
    )
    if raw.returncode != 0:
        raise SystemExit(raw.stderr.strip() or f"gh api failed for run {run_id}")
    j = json.loads(raw.stdout)
    return {
        "run_id": j["id"],
        "workflow": Path(j["path"]).name,
        "event": j["event"],
        "branch": j["head_branch"],
        "commit": j["head_sha"][:7],
        "conclusion": j["conclusion"],
        "started_at": j["run_started_at"],
    }


def upsert(doc: dict, entry: dict) -> dict:
    runs = [r for r in doc["runs"] if r["run_id"] != entry["run_id"]]
    runs.append(entry)
    doc["runs"] = sorted(runs, key=lambda r: r["started_at"])
    return doc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_id", type=int)
    parser.add_argument("--path", type=Path, default=RUNS_PATH)
    parser.add_argument("--target", default=None, help="warehouse target the run chose")
    parser.add_argument("--pushed", default=None, help="<hf repo>@<revision> the run pushed")
    parser.add_argument("--note", default=None)
    args = parser.parse_args(argv)
    doc = json.loads(args.path.read_text())
    entry = fetch(args.run_id)
    for key in ("target", "pushed", "note"):
        value = getattr(args, key)
        if value:
            entry[key] = value
    doc = upsert(doc, entry)
    args.path.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"recorded {entry['workflow']} run {entry['run_id']} ({entry['conclusion']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
