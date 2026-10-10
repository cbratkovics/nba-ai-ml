"""The README's dbt counts are rendered from the manifest and cannot drift (ADR-0022): this
parses the dbt project (no warehouse needed) and compares the rendering with the README."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from nba import config

sys.path.insert(0, str(config.REPO_ROOT / "scripts"))
import render_dbt_counts as rdc  # noqa: E402

pytest.importorskip("dbt.cli.main")


def _parse(target_path: Path) -> dict:
    script = Path(sys.executable).parent / "dbt"
    cmd = [
        *([str(script)] if script.exists() else [sys.executable, "-m", "dbt"]),
        "parse",
        "--project-dir",
        "dbt",
        "--profiles-dir",
        "dbt",
        "--target",
        "local",
        "--target-path",
        str(target_path),
        "--log-path",
        str(target_path / "logs"),
    ]
    env = {**os.environ, "DBT_TARGET": "local", "NBA_DUCKDB_PATH": str(target_path / "x.duckdb")}
    proc = subprocess.run(cmd, cwd=config.REPO_ROOT, env=env, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    return json.loads((target_path / "manifest.json").read_text())


def test_readme_counts_match_the_parsed_project(tmp_path: Path) -> None:
    manifest = _parse(tmp_path)
    c = rdc.counts(manifest)
    assert c["models"] == sum(c["layers"].values()), c
    assert c["tests"] == c["singular_tests"] + c["generic_tests"]
    readme = (config.REPO_ROOT / "README.md").read_text()
    assert rdc.current(readme) == rdc.render(c), (
        "run: dbt parse --project-dir dbt --profiles-dir dbt && python scripts/render_dbt_counts.py"
    )


def test_update_rewrites_only_between_the_markers(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text(f"before {rdc.START}old{rdc.END} after\n")
    assert rdc.update(readme, "new") is True
    assert readme.read_text() == f"before {rdc.START}new{rdc.END} after\n"
    assert rdc.update(readme, "new") is False
    readme.write_text("no markers\n")
    with pytest.raises(ValueError, match="markers"):
        rdc.current(readme.read_text())
