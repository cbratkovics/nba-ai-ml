"""Push and pull Parquet game logs and model files to Hugging Face Hub.

Parquet is never committed to git; the dataset repo is the source of truth.
The write token is read once, in `nba.config.hf_token()`.

Layout in the dataset repo:
    README.md                              dataset card
    game_logs/game_logs_<season>.parquet   one file per season

Usage:
    python -m nba.storage.hf push-dataset [--data-dir data/game_logs] \\
        [--card data/game_logs/README.md]
    python -m nba.storage.hf pull-dataset [--data-dir data/game_logs] [--revision SHA]
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download, snapshot_download

from nba import config
from nba.storage import local

DATASET_CARD_FILE = "README.md"


def _api(require_token: bool) -> HfApi:
    token = config.hf_token()
    if require_token and not token:
        raise RuntimeError("HF_TOKEN is not set; a write token is required to push")
    return HfApi(token=token)


def dataset_parquet_pattern() -> str:
    return f"{config.HF_DATASET_PREFIX}/{local.FILE_PREFIX}*.parquet"


def push_dataset(
    data_dir: Path = config.DATA_DIR,
    repo_id: str = config.HF_DATASET_REPO,
    message: str = "Update game logs",
    card_path: Path | None = None,
    files: list[Path] | None = None,
) -> str:
    """Upload the per-season Parquet files (and the dataset card, if given) in one commit.

    `files` restricts the upload to those Parquet files (default: every per-season file
    in data_dir). Returns the new commit sha.
    """
    files = local.list_parquet(data_dir) if files is None else list(files)
    if not files:
        raise FileNotFoundError(f"nothing to push: no Parquet files in {data_dir}")
    api = _api(require_token=True)
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
    operations = [
        CommitOperationAdd(
            path_in_repo=f"{config.HF_DATASET_PREFIX}/{p.name}", path_or_fileobj=str(p)
        )
        for p in files
    ]
    if card_path is not None:
        operations.append(
            CommitOperationAdd(path_in_repo=DATASET_CARD_FILE, path_or_fileobj=str(card_path))
        )
    info = api.create_commit(
        repo_id=repo_id, repo_type="dataset", operations=operations, commit_message=message
    )
    return info.oid


def fetch_dataset_card(repo_id: str = config.HF_DATASET_REPO) -> str:
    """Current README.md of the dataset repo."""
    path = hf_hub_download(repo_id, DATASET_CARD_FILE, repo_type="dataset", token=config.hf_token())
    return Path(path).read_text()


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
            allow_patterns=[dataset_parquet_pattern()],
            local_dir=tmp,
            token=config.hf_token(),
        )
        pulled = sorted((Path(tmp) / config.HF_DATASET_PREFIX).glob("*.parquet"))
        if not pulled:
            raise FileNotFoundError(f"no {dataset_parquet_pattern()} files in {repo_id}@{sha}")
        for src in pulled:
            shutil.copyfile(src, data_dir / src.name)
    (data_dir / local.REVISION_FILE).write_text(sha + "\n")
    return sha


def push_model(
    files: list[Path],
    repo_id: str = config.HF_MODEL_REPO,
    message: str = "Update model",
) -> str:
    """Upload model files (and a model card) to the model repo in one commit."""
    if not files:
        raise ValueError("no files given")
    api = _api(require_token=True)
    api.create_repo(repo_id, repo_type="model", exist_ok=True)
    operations = [CommitOperationAdd(path_in_repo=p.name, path_or_fileobj=str(p)) for p in files]
    info = api.create_commit(
        repo_id=repo_id, repo_type="model", operations=operations, commit_message=message
    )
    return info.oid


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p_push = sub.add_parser("push-dataset")
    p_push.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    p_push.add_argument("--message", default="Update game logs")
    p_push.add_argument("--card", type=Path, default=None, help="README.md to upload too")
    p_pull = sub.add_parser("pull-dataset")
    p_pull.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    p_pull.add_argument("--revision", default=None)
    args = parser.parse_args(argv)

    if args.command == "push-dataset":
        print(push_dataset(args.data_dir, message=args.message, card_path=args.card))
    elif args.command == "pull-dataset":
        sha = pull_dataset(args.data_dir, revision=args.revision)
        print(f"pulled {repo_files(args.data_dir)} at {sha}")


def repo_files(data_dir: Path) -> list[str]:
    return [p.name for p in local.list_parquet(data_dir)]


if __name__ == "__main__":
    main()
