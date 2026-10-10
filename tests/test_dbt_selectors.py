"""dbt/selectors.yml is the one definition of the nightly selection (ADR-0023): the Makefile
and nightly.yml use it by name, and it resolves to every node the nightly must rebuild."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import Counter

import pytest
import yaml

from nba import config

pytest.importorskip("dbt.cli.main")

ROOT = config.REPO_ROOT


def _ls(*args: str) -> list[dict]:
    script = sys.executable.rsplit("/", 1)[0] + "/dbt"
    cmd = [script, "ls", "--project-dir", "dbt", "--profiles-dir", "dbt", "--output", "json", *args]
    env = {**os.environ, "DBT_TARGET": "local"}
    proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    return [json.loads(line) for line in proc.stdout.splitlines() if line.startswith("{")]


def test_selector_is_declared_once_and_used_by_name() -> None:
    selectors = yaml.safe_load((ROOT / "dbt" / "selectors.yml").read_text())["selectors"]
    assert [s["name"] for s in selectors] == ["nightly"]
    makefile = (ROOT / "Makefile").read_text()
    nightly = (ROOT / ".github" / "workflows" / "nightly.yml").read_text()
    assert "--selector nightly" in makefile and "--selector nightly" in nightly
    # No hand-typed model list survives anywhere the selection is used.
    for text, where in ((makefile, "Makefile"), (nightly, "nightly.yml")):
        assert '--select "+slv_game_logs+"' not in text, where


def test_nightly_selector_covers_every_model_seed_and_snapshot() -> None:
    selected = _ls("--selector", "nightly")
    everything = _ls("--select", "fqn:*")
    kinds = Counter(n["resource_type"] for n in selected)
    assert kinds == Counter(n["resource_type"] for n in everything), kinds
    assert kinds["model"] >= 31 and kinds["seed"] >= 4 and kinds["snapshot"] == 1
    names = {n["name"] for n in selected}
    # The hand-typed list this replaced left slv_restatements out, so mart_restatement_lag
    # rebuilt against a stale parent on nightly runs (docs/AE_AUDIT.md, Phase 1).
    assert {"slv_restatements", "mart_restatement_lag", "seasons", "snp_player_team"} <= names
