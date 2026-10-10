"""The exposures close the lineage graph (ADR-0033): every site page has one, the Hugging Face
export exposure names exactly the exported gold models, and the analysis exposures name the
tables their scripts read. Parses the project (no warehouse needed), like test_dbt_counts."""

from __future__ import annotations

import re

import pytest

from nba import config
from tests.test_dbt_counts import _parse

pytest.importorskip("dbt.cli.main")

FRONTEND_APP = config.REPO_ROOT / "frontend" / "app"
SITE = "https://nba-ai-ml.vercel.app"
PRIVATE = {"fct_paper_bet", "slv_lines", "brz_lines", "lines", "captures"}


@pytest.fixture(scope="module")
def manifest(tmp_path_factory) -> dict:
    return _parse(tmp_path_factory.mktemp("parse"))


def _exposures(manifest: dict) -> dict[str, dict]:
    return {e["name"]: e for e in manifest["exposures"].values()}


def _deps(manifest: dict, exposure: dict) -> set[str]:
    """The relation names an exposure depends on: model aliases (fct_decision_policy, not the
    versioned node name) and source table names."""
    names = set()
    for uid in exposure["depends_on"]["nodes"]:
        node = manifest["nodes"].get(uid) or manifest["sources"].get(uid)
        assert node is not None, uid
        names.add(node.get("alias") or node["name"])
    return names


def _exported_models(manifest: dict) -> set[str]:
    return {
        n["alias"]
        for n in manifest["nodes"].values()
        if n["resource_type"] == "model"
        and "gold" in n["tags"]
        and (n["config"].get("meta") or {}).get("export", True) is not False
    }


def _routes() -> dict[str, str]:
    """site route -> page.tsx text."""
    routes = {}
    for page in FRONTEND_APP.rglob("page.tsx"):
        rel = page.parent.relative_to(FRONTEND_APP).as_posix()
        routes["/" if rel == "." else f"/{rel}"] = page.read_text()
    return routes


def _dashboards(manifest: dict) -> dict[str, dict]:
    return {
        e["url"].removeprefix(SITE) or "/": e
        for e in _exposures(manifest).values()
        if e["type"] == "dashboard"
    }


def test_every_site_page_has_a_dashboard_exposure(manifest: dict) -> None:
    dashboards = _dashboards(manifest)
    routes = set(_routes())
    assert set(dashboards) == routes, set(dashboards) ^ routes
    for route, e in dashboards.items():
        assert e["owner"]["name"] and e["description"].strip(), route
        assert _deps(manifest, e), f"{route} depends on nothing"


def test_the_export_exposure_names_exactly_the_exported_gold_models(manifest: dict) -> None:
    exported = _exported_models(manifest)
    listed = _deps(manifest, _exposures(manifest)["hf_gold_export"])
    assert listed == exported, listed ^ exported
    assert "fct_paper_bet" not in listed and "int_player_game_features" not in listed


def test_pages_read_the_marts_their_exposures_name(manifest: dict) -> None:
    """A page's exposure lists every gold mart the page fetches (getGoldMart('<mart>'))."""
    dashboards = _dashboards(manifest)
    for route, text in _routes().items():
        fetched = set(re.findall(r"getGoldMart<\w+>\('(\w+)'\)", text))
        listed = _deps(manifest, dashboards[route])
        assert fetched <= listed, (route, fetched - listed)
        # Every mart a page fetches is exported as JSON.
        for mart in fetched:
            node = next(n for n in manifest["nodes"].values() if n.get("alias") == mart)
            assert (node["config"].get("meta") or {}).get("export_json") is True, mart


def test_analysis_exposures_name_the_tables_their_scripts_read(manifest: dict) -> None:
    exposures = _exposures(manifest)
    checks = {
        "policy_evaluate": config.REPO_ROOT / "nba" / "decisions" / "evaluate.py",
        "drift_reference_and_calibration": config.REPO_ROOT / "nba" / "drift" / "reference.py",
    }
    for name, path in checks.items():
        text = path.read_text()
        for table in _deps(manifest, exposures[name]):
            assert f"gold.{table}" in text, (name, table)
    tools = (config.REPO_ROOT / "nba" / "agent" / "tools.py").read_text()
    models = {n["alias"] for n in manifest["nodes"].values() if n["resource_type"] == "model"}
    for mart in _deps(manifest, exposures["agent_nightly_brief"]) & models:
        assert f'"{mart}"' in tools, mart


def test_exposure_count_is_rendered_in_the_readme(manifest: dict) -> None:
    n = len(manifest["exposures"])
    assert n >= 11
    readme = (config.REPO_ROOT / "README.md").read_text()
    assert f"{n} exposures" in readme, "run: dbt parse ... && python scripts/render_dbt_counts.py"


def test_private_tables_are_not_behind_a_public_page(manifest: dict) -> None:
    for e in _exposures(manifest).values():
        if e["type"] == "dashboard":
            assert _deps(manifest, e).isdisjoint(PRIVATE), e["name"]
