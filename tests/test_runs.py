"""reports/runs.json is the run-evidence artifact (ADR-0022): every GitHub Actions run id
cited in the README, the docs or a workflow header must be recorded there with its workflow,
commit, conclusion and start time, so "ran" always has a run id behind it."""

from __future__ import annotations

import json
import re
from datetime import datetime

from nba import config

ROOT = config.REPO_ROOT
RUNS = ROOT / "reports" / "runs.json"
# Actions run ids on this repository are eleven digits starting with 3 (2026).
RUN_ID = re.compile(r"(?<![\d.])(3[4-9]\d{9})(?![\d.])")
CITING_FILES = [
    ROOT / "README.md",
    ROOT / "AUDIT.md",
    *sorted((ROOT / "docs").glob("*.md")),
    *sorted((ROOT / ".github" / "workflows").glob("*.yml")),
    ROOT / "dbt" / "models" / "overview.md",
]
REQUIRED = {"run_id", "workflow", "event", "branch", "commit", "conclusion", "started_at"}
WORKFLOWS = {p.name for p in (ROOT / ".github" / "workflows").glob("*.yml")}


def _doc() -> dict:
    return json.loads(RUNS.read_text())


def test_runs_file_is_well_formed() -> None:
    doc = _doc()
    ids = [r["run_id"] for r in doc["runs"]]
    assert len(ids) == len(set(ids)), "duplicate run ids"
    assert ids == [r["run_id"] for r in sorted(doc["runs"], key=lambda r: r["started_at"])]
    for r in doc["runs"]:
        assert REQUIRED <= set(r), r
        assert r["workflow"] in WORKFLOWS, r["workflow"]
        assert r["event"] in {"schedule", "workflow_dispatch", "push", "pull_request"}
        assert r["conclusion"] in {"success", "failure", "cancelled"}
        assert re.fullmatch(r"[0-9a-f]{7}", r["commit"]), r["commit"]
        datetime.fromisoformat(r["started_at"].replace("Z", "+00:00"))
        if "pushed" in r:
            assert re.fullmatch(r"cbratkovics/[\w-]+@[0-9a-f]{8}", r["pushed"]), r["pushed"]
    for c in doc["hf_commits"]:
        assert {"repo", "repo_type", "revision", "date", "title"} <= set(c), c
        assert c["repo_type"] in {"dataset", "model"}
        if "run_id" in c:
            assert c["run_id"] in set(ids), f"hf commit {c['revision']} cites unknown run"


def test_every_cited_run_id_is_recorded() -> None:
    recorded = {r["run_id"] for r in _doc()["runs"]}
    missing: dict[str, set[int]] = {}
    for path in CITING_FILES:
        if not path.exists():
            continue
        cited = {int(m) for m in RUN_ID.findall(path.read_text())}
        gap = cited - recorded
        if gap:
            missing[str(path.relative_to(ROOT))] = gap
    assert not missing, f"run ids cited but not in reports/runs.json: {missing}"


def test_the_runs_the_status_claims_rest_on_are_present() -> None:
    """The ids the README's status table and ADR-0022 name, by workflow and conclusion."""
    runs = {r["run_id"]: r for r in _doc()["runs"]}
    warehouse = [r for r in runs.values() if r["workflow"] == "warehouse.yml"]
    assert len(warehouse) >= 4 and all(r["conclusion"] == "success" for r in warehouse)
    assert all(r.get("target") == "motherduck" for r in warehouse), "target must be recorded"
    assert runs[35048092357]["pushed"] == "cbratkovics/nba-game-logs@66404be0"
    # Live nightly runs only: a rehearsal dispatch (ADR-0028) is recorded with "rehearsal" in
    # its note and may fail without touching the claim that every live night succeeded.
    nightly = [
        r
        for r in runs.values()
        if r["workflow"] == "nightly.yml" and "rehearsal" not in r.get("note", "")
    ]
    assert len(nightly) >= 28 and all(r["conclusion"] == "success" for r in nightly)
    assert any(
        "rehearsal" in r.get("note", "") and r["conclusion"] == "success" for r in runs.values()
    )
    assert runs[34666354516]["workflow"] == "train.yml"
    assert runs[34668262142]["workflow"] == "replay.yml"
