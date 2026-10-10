#!/usr/bin/env python
"""Render the README's dbt counts from dbt/target/manifest.json, never typed by hand.

The README carries the counts between `<!-- dbt-counts -->` markers (ADR-0022, the same
mechanism as the agent pass-rate row). `--check` exits 1 when the README differs from the
manifest; tests/test_dbt_counts.py runs `dbt parse` and does the same, so a model, test or
seed added without re-rendering fails CI.

    python scripts/render_dbt_counts.py [--manifest dbt/target/manifest.json]
                                        [--readme README.md] [--check]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

START = "<!-- dbt-counts -->"
END = "<!-- /dbt-counts -->"
LAYERS = ("bronze", "silver", "gold")


def counts(manifest: dict) -> dict:
    nodes = manifest["nodes"].values()
    models = [n for n in nodes if n["resource_type"] == "model"]
    tests = [n for n in nodes if n["resource_type"] == "test"]
    by_layer = Counter(n["config"].get("schema") or n["schema"] for n in models)
    documented = sum(
        len(n.get("columns", {}))
        for n in nodes
        if n["resource_type"] in ("model", "seed", "snapshot")
    )
    return {
        "models": len(models),
        "layers": {layer: by_layer.get(layer, 0) for layer in LAYERS},
        "incremental": sum(1 for n in models if n["config"]["materialized"] == "incremental"),
        "contracted": sum(1 for n in models if (n["config"].get("contract") or {}).get("enforced")),
        "seeds": sum(1 for n in nodes if n["resource_type"] == "seed"),
        "snapshots": sum(1 for n in nodes if n["resource_type"] == "snapshot"),
        "tests": len(tests),
        "singular_tests": sum(1 for t in tests if t.get("test_metadata") is None),
        "generic_tests": sum(1 for t in tests if t.get("test_metadata") is not None),
        "sources": len(manifest.get("sources", {})),
        "exposures": len(manifest.get("exposures", {})),
        "unit_tests": len(manifest.get("unit_tests", {})),
        "documented_columns": documented,
    }


def render(c: dict) -> str:
    layers = ", ".join(f"{c['layers'][layer]} {layer}" for layer in LAYERS)
    return (
        f"{c['models']} models ({layers}; {c['incremental']} incremental, "
        f"{c['contracted']} with enforced contracts), {c['seeds']} seeds, "
        f"{c['snapshots']} snapshot, {c['tests']} data tests ({c['singular_tests']} singular, "
        f"{c['generic_tests']} generic), {c['unit_tests']} unit tests, {c['sources']} sources, "
        f"{c['exposures']} exposures, {c['documented_columns']} documented columns"
    )


def current(readme_text: str) -> str:
    start, end = readme_text.find(START), readme_text.find(END)
    if start < 0 or end < 0:
        raise ValueError(f"README has no {START} … {END} markers")
    return readme_text[start + len(START) : end]


def update(readme: Path, line: str) -> bool:
    text = readme.read_text()
    old = current(text)
    if old == line:
        return False
    readme.write_text(text.replace(f"{START}{old}{END}", f"{START}{line}{END}"))
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--manifest", type=Path, default=Path("dbt") / "target" / "manifest.json")
    parser.add_argument("--readme", type=Path, default=Path("README.md"))
    parser.add_argument("--check", action="store_true", help="exit 1 instead of rewriting")
    args = parser.parse_args(argv)
    line = render(counts(json.loads(args.manifest.read_text())))
    if args.check:
        if current(args.readme.read_text()) != line:
            print(f"README dbt counts are stale; expected:\n{line}", file=sys.stderr)
            return 1
        print("ok: README dbt counts match the manifest")
        return 0
    changed = update(args.readme, line)
    print(("updated: " if changed else "unchanged: ") + line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
