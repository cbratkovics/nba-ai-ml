"""Record the SHA-256 of every file behind the published numbers.

`reports/metrics.json` names the dataset by Hugging Face commit and the model by the git
commit of the training code, but neither is a hash of file content: a re-push of the same
revision, a rewritten parquet file, or a replaced model file would not be detectable. This
module writes `reports/provenance_<dataset revision[:8]>.json` with:

- the SHA-256, byte size and row count of every per-season parquet file at the pinned
  dataset revision, checked against the LFS hash Hugging Face reports for the same file;
- the SHA-256 and size of every model file and of `metrics.json` at the pinned model
  revision, with a flag saying whether that `metrics.json` is byte-identical to the
  committed `reports/metrics.json`;
- the SHA-256 of the committed reports;
- the SHA-256 and revision of the replay products the site reads from the dataset repo's
  `main` branch, with a flag saying whether the committed replay report is the same run.

Usage:
    python -m nba.storage.provenance [--out reports/provenance_b20b5601.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from nba import config

REPORT_FILES: tuple[str, ...] = (
    "reports/metrics.json",
    "reports/replay_2025-26.json",
    "reports/replay_all_rows_2025-26.json",
    "reports/agent_evals.json",
    "reports/agent_golden.json",
    "reports/agent_pass_rates.json",
)
HUB_PRODUCTS: tuple[str, ...] = (
    "replay/2025-26/replay.json",
    "replay/2025-26/daily_mae.json",
)
MODEL_FILES: tuple[str, ...] = ("lgbm_pts.txt", "lgbm_reb.txt", "lgbm_ast.txt", "metrics.json")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def provenance_path(dataset_revision: str = config.DATASET_REVISION) -> Path:
    return config.REPORTS_DIR / f"provenance_{dataset_revision[:8]}.json"


def _parquet_rows(path: Path) -> int:
    import pyarrow.parquet as pq

    return int(pq.ParquetFile(path).metadata.num_rows)


def build(
    download: Callable[[str, str, str, str], Path],
    lfs_sha256: Callable[[str, str, str], dict[str, str]],
    repo_sha: Callable[[str, str, str], str],
    repo_root: Path = config.REPO_ROOT,
    dataset_revision: str = config.DATASET_REVISION,
    model_revision: str = config.MODEL_REVISION,
    git_sha: str | None = None,
) -> dict[str, Any]:
    """Assemble the provenance report.

    `download(repo_id, repo_type, path, revision)` returns a local path;
    `lfs_sha256(repo_id, repo_type, revision)` maps repo paths to the LFS SHA-256 Hugging
    Face reports (only for LFS files); `repo_sha(repo_id, repo_type, revision)` resolves a
    branch name to a commit. All three are injected so the report can be built in tests.
    """
    dataset_files: dict[str, Any] = {}
    lfs = lfs_sha256(config.HF_DATASET_REPO, "dataset", dataset_revision)
    for season in config.SEASONS:
        rel = f"{config.HF_DATASET_PREFIX}/game_logs_{season}.parquet"
        local_path = download(config.HF_DATASET_REPO, "dataset", rel, dataset_revision)
        digest = sha256_of(local_path)
        dataset_files[rel] = {
            "sha256": digest,
            "bytes": local_path.stat().st_size,
            "rows": _parquet_rows(local_path),
            "matches_hub_lfs_sha256": lfs.get(rel) == digest if rel in lfs else None,
        }

    model_files: dict[str, Any] = {}
    for name in MODEL_FILES:
        local_path = download(config.HF_MODEL_REPO, "model", name, model_revision)
        model_files[name] = {"sha256": sha256_of(local_path), "bytes": local_path.stat().st_size}

    reports: dict[str, Any] = {}
    for rel in REPORT_FILES:
        path = repo_root / rel
        if path.exists():
            reports[rel] = {"sha256": sha256_of(path), "bytes": path.stat().st_size}

    main_sha = repo_sha(config.HF_DATASET_REPO, "dataset", "main")
    hub_products: dict[str, Any] = {}
    for rel in HUB_PRODUCTS:
        local_path = download(config.HF_DATASET_REPO, "dataset", rel, main_sha)
        hub_products[rel] = {
            "revision": main_sha,
            "sha256": sha256_of(local_path),
            "bytes": local_path.stat().st_size,
        }

    metrics_committed = reports.get("reports/metrics.json", {}).get("sha256")
    replay_committed = reports.get("reports/replay_2025-26.json", {}).get("sha256")
    all_rows_committed = reports.get("reports/replay_all_rows_2025-26.json", {}).get("sha256")
    checks = {
        "model_repo_metrics_json_equals_committed": (
            model_files["metrics.json"]["sha256"] == metrics_committed
        ),
        "hub_replay_json_equals_committed_report": (
            hub_products["replay/2025-26/replay.json"]["sha256"] == replay_committed
        ),
        "hub_daily_mae_equals_committed_all_rows_report": (
            hub_products["replay/2025-26/daily_mae.json"]["sha256"] == all_rows_committed
        ),
        "all_parquet_match_hub_lfs_sha256": all(
            v["matches_hub_lfs_sha256"] for v in dataset_files.values()
        ),
    }
    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_sha": git_sha,
        "model_identity": config.model_identity(),
        "dataset": {
            "repo": config.HF_DATASET_REPO,
            "revision": dataset_revision,
            "files": dataset_files,
        },
        "model": {
            "repo": config.HF_MODEL_REPO,
            "revision": model_revision,
            "commit": config.MODEL_COMMIT,
            "files": model_files,
        },
        "reports": reports,
        "hub_products": hub_products,
        "checks": checks,
    }


def _hub_download(repo_id: str, repo_type: str, path: str, revision: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id, path, repo_type=repo_type, revision=revision, token=config.hf_token()
        )
    )


def _hub_lfs_sha256(repo_id: str, repo_type: str, revision: str) -> dict[str, str]:
    from huggingface_hub import HfApi

    info = HfApi(token=config.hf_token()).repo_info(
        repo_id, repo_type=repo_type, revision=revision, files_metadata=True
    )
    return {s.rfilename: s.lfs.sha256 for s in info.siblings if s.lfs is not None}


def _hub_repo_sha(repo_id: str, repo_type: str, revision: str) -> str:
    from huggingface_hub import HfApi

    return (
        HfApi(token=config.hf_token())
        .repo_info(repo_id, repo_type=repo_type, revision=revision)
        .sha
    )


def main(argv: list[str] | None = None) -> int:
    from nba.models import evaluate

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    out = args.out or provenance_path()
    report = build(_hub_download, _hub_lfs_sha256, _hub_repo_sha, git_sha=evaluate.git_sha())
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    for k, v in report["checks"].items():
        print(f"PROVENANCE {k}: {v}")
    print(f"PROVENANCE wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
