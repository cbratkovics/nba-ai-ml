"""Push the exported gold marts (dbt run-operation export_gold) to the dataset repo.

Files land under gold/<alias>.parquet plus gold/_export_manifest.json, so a consumer can read
a mart the same way it reads every other product: by public URL at resolve/main. Today the
analyst tools are the only reader (nba/agent/tools.py); the site reads the product files and
will read small JSON exports of the marts from Phase 3 (docs/AE_AUDIT.md). Nothing else in the
repo is touched. Run by warehouse.yml after every full refresh and by nightly.yml after a
night that built the warehouse (ADR-0022).

Usage:
    python -m nba.warehouse.publish [--export-dir data/warehouse/export] [--dry-run]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from nba import config

DEFAULT_EXPORT_DIR = Path("data") / "warehouse" / "export"


def export_files(export_dir: Path) -> list[Path]:
    files = sorted(p for p in export_dir.glob("*.parquet") if p.is_file())
    manifest = export_dir / "_export_manifest.json"
    if manifest.exists():
        files.append(manifest)
    return files


def publish(export_dir: Path = DEFAULT_EXPORT_DIR, dry_run: bool = False) -> str | None:
    from nba.storage import hf

    files = export_files(export_dir)
    if not files:
        raise FileNotFoundError(f"no exported marts in {export_dir}; run export_gold first")
    if dry_run:
        for f in files:
            print(f"PUBLISH would upload {config.HF_GOLD_PREFIX}/{f.name}")
        return None
    return hf.push_gold(files)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT_DIR)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    sha = publish(args.export_dir, dry_run=args.dry_run)
    if sha:
        print(f"PUBLISH pushed gold marts to {config.HF_DATASET_REPO} at {sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
