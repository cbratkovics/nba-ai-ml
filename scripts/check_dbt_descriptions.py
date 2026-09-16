#!/usr/bin/env python
"""Fail when any dbt model, source table, seed or model column lacks a description.

Reads dbt/target/manifest.json (declared descriptions) and dbt/target/catalog.json (the
columns that actually exist in the built warehouse), so a column added to a model's SQL
without a YAML entry is caught too, not only a YAML entry with an empty description. Run
after ``dbt docs generate``:

    python scripts/check_dbt_descriptions.py [--target-dir dbt/target]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _blank(text: str | None) -> bool:
    return not (text or "").strip()


def find_gaps(manifest: dict, catalog: dict) -> list[str]:
    gaps: list[str] = []
    nodes = manifest.get("nodes", {})
    for uid, node in sorted(nodes.items()):
        if node.get("resource_type") not in ("model", "seed", "snapshot"):
            continue
        name = node["name"]
        if _blank(node.get("description")):
            gaps.append(f"{node['resource_type']} {name}: no description")
        declared = node.get("columns", {})
        actual = catalog.get("nodes", {}).get(uid, {}).get("columns", {})
        if not actual:
            gaps.append(
                f"{node['resource_type']} {name}: not in catalog.json (built before docs generate?)"
            )
        for col in actual:
            if col not in declared:
                gaps.append(f"{name}: column {col} exists but is not documented in YAML")
            elif _blank(declared[col].get("description")):
                gaps.append(f"{name}: column {col} has no description")
        for col in declared:
            if actual and col not in actual:
                gaps.append(f"{name}: YAML documents column {col} which the relation does not have")
    for _uid, src in sorted(manifest.get("sources", {}).items()):
        if _blank(src.get("description")):
            gaps.append(f"source {src['source_name']}.{src['name']}: no description")
    return gaps


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--target-dir", default="dbt/target", type=Path)
    args = parser.parse_args(argv)
    manifest_path = args.target_dir / "manifest.json"
    catalog_path = args.target_dir / "catalog.json"
    for path in (manifest_path, catalog_path):
        if not path.exists():
            print(f"missing {path}; run `dbt docs generate` first", file=sys.stderr)
            return 2
    manifest = json.loads(manifest_path.read_text())
    catalog = json.loads(catalog_path.read_text())
    gaps = find_gaps(manifest, catalog)
    n_models = sum(1 for n in manifest["nodes"].values() if n["resource_type"] == "model")
    n_cols = sum(len(n["columns"]) for n in catalog["nodes"].values())
    if gaps:
        print(f"{len(gaps)} description gap(s) across {n_models} models:")
        for gap in gaps:
            print(f"  - {gap}")
        return 1
    print(f"ok: every model, seed, snapshot, column ({n_cols}) and source has a description")
    return 0


if __name__ == "__main__":
    sys.exit(main())
