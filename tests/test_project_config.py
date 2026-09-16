"""The dbt project's vars mirror nba/config.py and the feature module; the warehouse cannot
drift from the pipeline's identity."""

import subprocess
from pathlib import Path

import yaml

from nba import config
from nba.features import asof

DBT_PROJECT = config.REPO_ROOT / "dbt" / "dbt_project.yml"


def _vars() -> dict:
    return yaml.safe_load(DBT_PROJECT.read_text())["vars"]


def test_dbt_vars_mirror_config() -> None:
    v = _vars()
    assert v["model_revision"] == config.MODEL_REVISION
    assert v["model_commit"] == config.MODEL_COMMIT
    assert v["dataset_revision"] == config.DATASET_REVISION
    assert v["feature_version"] == asof.FEATURE_VERSION
    assert v["holdout_season"] == config.HOLDOUT_SEASON
    assert v["seasons"] == list(config.SEASONS)
    assert v["targets"] == list(config.TARGETS)
    assert v["min_minutes"] == config.MIN_MINUTES
    # conftest relaxes config.FULL_SEASON_GAMES for the fixture; compare with the real value.
    assert v["full_season_games"] == 30 * 82 // 2
    assert v["lookback_days"] == 2 * config.DAILY_LOOKBACK_DAYS


def test_constraints_pin_the_dev_extra() -> None:
    pins = {
        line.split("==")[0]: line.split("==")[1]
        for line in (config.REPO_ROOT / "constraints.txt").read_text().splitlines()
        if "==" in line and not line.startswith("#")
    }
    pyproject = (config.REPO_ROOT / "pyproject.toml").read_text()
    for name, version in pins.items():
        assert f'"{name}=={version}"' in pyproject, name


def test_seeds_are_tracked_despite_the_csv_ignore() -> None:
    ignore = (config.REPO_ROOT / ".gitignore").read_text()
    assert "!dbt/seeds/*.csv" in ignore
    assert list((config.REPO_ROOT / "dbt" / "seeds").glob("*.csv"))
    assert Path(config.REPO_ROOT / "dbt" / "seeds" / "known_stat_exceptions.csv").exists()


def test_no_warehouse_file_is_git_ignored() -> None:
    """A global ignore (*.sql, *.csv, *.json) must never silently drop a warehouse file."""
    root = config.REPO_ROOT
    files = [
        p.relative_to(root).as_posix()
        for d in ("dbt", "scripts")
        for p in (root / d).rglob("*")
        if p.is_file() and not any(part in ("target", "logs", "dbt_packages") for part in p.parts)
    ]
    assert files
    proc = subprocess.run(
        ["git", "check-ignore", "--no-index", *files], cwd=root, capture_output=True, text=True
    )
    assert proc.stdout.strip() == "", f"ignored files:\n{proc.stdout}"
