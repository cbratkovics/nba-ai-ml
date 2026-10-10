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
    assert "seasons" not in v, "the season list is the seasons seed, not a var (ADR-0024)"
    assert v["targets"] == list(config.TARGETS)
    assert v["min_minutes"] == config.MIN_MINUTES
    # conftest relaxes config.FULL_SEASON_GAMES for the fixture; compare with the real value.
    assert v["full_season_games"] == 30 * 82 // 2
    assert v["lookback_days"] == 2 * config.DAILY_LOOKBACK_DAYS


def test_ledger_vars_mirror_the_rule() -> None:
    """fct_paper_bet applies the rule nba/ledger/rule.py carries; the hash pins the whole text."""
    from nba.ledger import rule

    v = _vars()
    assert v["ledger_rule_version"] == rule.RULE["version"]
    assert v["ledger_rule_hash"] == rule.rule_hash()
    assert (
        v["ledger_edge_threshold_pts"] == rule.RULE["edge"]["threshold_by_market"]["player_points"]
    )
    e = rule.RULE["eligibility"]
    assert v["ledger_min_prior_games_season"] == e["min_prior_games_season"]
    assert v["ledger_min_minutes_mean_last10"] == e["min_minutes_mean_last10"]
    assert v["ledger_max_input_staleness_days"] == e["max_input_staleness_days"]
    assert v["ledger_unit"] == rule.RULE["stake"]["units"]


def test_seasons_seed_is_the_one_season_list() -> None:
    """The seed holds every season the pipeline knows (config.SEASONS) plus the current one by
    the October rule, each labelled by the same rule the package uses (ADR-0024)."""
    import csv
    from datetime import date

    from nba.ingest.schedule import season_for_date

    rows = list(csv.DictReader((config.REPO_ROOT / "dbt" / "seeds" / "seasons.csv").open()))
    seasons = [r["season"] for r in rows]
    assert seasons == sorted(seasons) and len(set(seasons)) == len(seasons)
    assert set(config.SEASONS) <= set(seasons)
    assert season_for_date(date.today()) in seasons, "add this season to dbt/seeds/seasons.csv"
    for r in rows:
        assert r["season"] == f"{r['start_year']}-{(int(r['start_year']) + 1) % 100:02d}"
        assert r["status"] in ("complete", "current", "upcoming")
    assert [r["season"] for r in rows if r["status"] == "complete"] == list(config.SEASONS)


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
        if p.is_file()
        and not any(part in ("target", "logs", "dbt_packages", "__pycache__") for part in p.parts)
        and p.name != ".user.yml"  # dbt's per-machine usage id: ignored on purpose
    ]
    assert files
    proc = subprocess.run(
        ["git", "check-ignore", "--no-index", *files], cwd=root, capture_output=True, text=True
    )
    assert proc.stdout.strip() == "", f"ignored files:\n{proc.stdout}"
