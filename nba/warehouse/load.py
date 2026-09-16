"""Load the warehouse's source files from Hugging Face into data/warehouse/.

Bronze reads files, never the Kaggle CSV: the five per-season game-log parquet files at a
pinned dataset revision, the nightly products (predictions/, residuals/, daily_reports/),
and the replay residuals for the holdout season. Every family lands under one root that the
dbt sources point at (var `warehouse_root`), and `load_manifest.json` records the revision,
file counts and load time so the warehouse can say which snapshot it was built from.

The committed reports (reports/*.json) are read by dbt in place; nothing copies them.

Usage:
    python -m nba.warehouse.load [--root data/warehouse] [--revision main|<sha>]
                                 [--season 2025-26] [--offline]
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from nba import config

DEFAULT_ROOT = Path("data") / "warehouse"
MANIFEST_FILE = "load_manifest.json"
# Product families copied from the dataset repo: (folder in the repo, glob inside it).
PRODUCT_FAMILIES: tuple[tuple[str, str], ...] = (
    (config.HF_PREDICTIONS_PREFIX, "*.parquet"),
    (config.HF_RESIDUALS_PREFIX, "*.parquet"),
    (config.HF_DAILY_REPORTS_PREFIX, "*.json"),
    (config.HF_DRIFT_PREFIX, "*.json"),
)


def replay_residuals_prefix(season: str) -> str:
    return f"{config.HF_REPLAY_PREFIX}/{season}/residuals"


def _count(root: Path, folder: str, pattern: str) -> int:
    d = root / folder
    return len(list(d.glob(pattern))) if d.is_dir() else 0


def write_manifest(
    root: Path, revision: str | None, season: str, loaded_at: str | None = None
) -> dict[str, Any]:
    families = {
        config.HF_DATASET_PREFIX: _count(root, config.HF_DATASET_PREFIX, "game_logs_*.parquet"),
        **{folder: _count(root, folder, pattern) for folder, pattern in PRODUCT_FAMILIES},
        replay_residuals_prefix(season): _count(root, replay_residuals_prefix(season), "*.parquet"),
    }
    manifest = {
        "loaded_at": loaded_at or datetime.now(UTC).isoformat(timespec="seconds"),
        "dataset_repo": config.HF_DATASET_REPO,
        "dataset_revision": revision,
        "replay_season": season,
        "model_revision": config.MODEL_REVISION,
        "model_commit": config.MODEL_COMMIT,
        "families": families,
        "root": root.as_posix(),
    }
    root.mkdir(parents=True, exist_ok=True)
    # export_gold copies the gold marts here; DuckDB's COPY does not create directories.
    (root / "export").mkdir(exist_ok=True)
    (root / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load(root: Path = DEFAULT_ROOT, revision: str | None = None, season: str | None = None) -> dict:
    """Download every source family at one dataset-repo revision. Returns the manifest."""
    from huggingface_hub import HfApi, snapshot_download

    from nba.storage import hf

    season = season or config.HOLDOUT_SEASON
    api = HfApi(token=config.hf_token())
    sha = api.dataset_info(config.HF_DATASET_REPO, revision=revision).sha
    root.mkdir(parents=True, exist_ok=True)
    hf.pull_dataset(root / config.HF_DATASET_PREFIX, revision=sha)
    patterns = [f"{folder}/{pattern}" for folder, pattern in PRODUCT_FAMILIES]
    patterns.append(f"{replay_residuals_prefix(season)}/*.parquet")
    with tempfile.TemporaryDirectory() as tmp:
        snapshot_download(
            config.HF_DATASET_REPO,
            repo_type="dataset",
            revision=sha,
            allow_patterns=patterns,
            local_dir=tmp,
            token=config.hf_token(),
        )
        for folder in [f for f, _ in PRODUCT_FAMILIES] + [replay_residuals_prefix(season)]:
            src = Path(tmp) / folder
            dest = root / folder
            if dest.exists():
                shutil.rmtree(dest)
            dest.mkdir(parents=True, exist_ok=True)
            if src.is_dir():
                for f in sorted(src.iterdir()):
                    if f.is_file():
                        shutil.copyfile(f, dest / f.name)
    return write_manifest(root, sha, season)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--revision", default=None, help="dataset-repo revision (default: main)")
    parser.add_argument("--season", default=config.HOLDOUT_SEASON, help="replay season to load")
    parser.add_argument(
        "--offline", action="store_true", help="only rewrite the manifest for files already present"
    )
    args = parser.parse_args(argv)
    if args.offline:
        existing = args.root / MANIFEST_FILE
        rev = json.loads(existing.read_text())["dataset_revision"] if existing.exists() else None
        manifest = write_manifest(args.root, rev, args.season)
    else:
        manifest = load(args.root, args.revision, args.season)
    for family, n in manifest["families"].items():
        print(f"LOAD {family}: {n} files")
    print(f"LOAD dataset revision {manifest['dataset_revision']} -> {args.root / MANIFEST_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
