"""Push and pull Parquet game logs and model files to Hugging Face Hub.

Parquet is never committed to git; the dataset repo is the source of truth.
The write token is read once, in `nba.config.hf_token()`.

Usage:
    python -m nba.storage.hf push-dataset [--data-dir data/game_logs]
    python -m nba.storage.hf pull-dataset [--data-dir data/game_logs] [--revision SHA]
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

from nba import config
from nba.storage import local


def _api(require_token: bool) -> HfApi:
    token = config.hf_token()
    if require_token and not token:
        raise RuntimeError("HF_TOKEN is not set; a write token is required to push")
    return HfApi(token=token)


def push_dataset(
    data_dir: Path = config.DATA_DIR,
    repo_id: str = config.HF_DATASET_REPO,
    message: str = "Update game logs",
) -> str:
    """Upload every per-season Parquet file. Returns the new commit sha."""
    files = local.list_parquet(data_dir)
    if not files:
        raise FileNotFoundError(f"nothing to push: no Parquet files in {data_dir}")
    api = _api(require_token=True)
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
    info = api.upload_folder(
        folder_path=str(data_dir),
        repo_id=repo_id,
        repo_type="dataset",
        path_in_repo=config.HF_DATASET_PREFIX,
        allow_patterns=[f"{local.FILE_PREFIX}*.parquet"],
        commit_message=message,
    )
    return info.oid


def pull_dataset(
    data_dir: Path = config.DATA_DIR,
    repo_id: str = config.HF_DATASET_REPO,
    revision: str | None = None,
) -> str:
    """Download the per-season Parquet files into data_dir. Returns the commit sha.

    The sha is also written to `data_dir/.hf_revision` so later runs can record
    which dataset version they used.
    """
    api = _api(require_token=False)
    sha = api.dataset_info(repo_id, revision=revision).sha
    data_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        snapshot_download(
            repo_id,
            repo_type="dataset",
            revision=sha,
            allow_patterns=[f"{config.HF_DATASET_PREFIX}/{local.FILE_PREFIX}*.parquet"],
            local_dir=tmp,
            token=config.hf_token(),
        )
        for src in sorted((Path(tmp) / config.HF_DATASET_PREFIX).glob("*.parquet")):
            shutil.copyfile(src, data_dir / src.name)
    (data_dir / local.REVISION_FILE).write_text(sha + "\n")
    return sha


def push_model(
    files: list[Path],
    repo_id: str = config.HF_MODEL_REPO,
    message: str = "Update model",
) -> str:
    """Upload model files (and a model card) to the model repo. Returns the commit sha."""
    api = _api(require_token=True)
    api.create_repo(repo_id, repo_type="model", exist_ok=True)
    operations = [(p, p.name) for p in files]
    sha = None
    for path, name in operations:
        info = api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=name,
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"{message}: {name}",
        )
        sha = info.oid
    if sha is None:
        raise ValueError("no files given")
    return sha


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p_push = sub.add_parser("push-dataset")
    p_push.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    p_push.add_argument("--message", default="Update game logs")
    p_pull = sub.add_parser("pull-dataset")
    p_pull.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    p_pull.add_argument("--revision", default=None)
    args = parser.parse_args(argv)

    if args.command == "push-dataset":
        print(push_dataset(args.data_dir, message=args.message))
    elif args.command == "pull-dataset":
        print(pull_dataset(args.data_dir, revision=args.revision))


if __name__ == "__main__":
    main()
