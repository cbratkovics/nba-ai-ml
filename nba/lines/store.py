"""The private lines repo: raw sportsbook lines, capture records, the usage log and the
recommendations live in a private Hugging Face dataset repo (HF_LINES_REPO), never in git and
never in the public dataset repo (ADR-0031; The Odds API terms permit storing the data and
publishing values derived from it, not redistributing it).

Layout:
    captures/<date>.json            the decision to capture, written before any priced call,
                                    then completed with the outcome
    lines/<date>.parquet            API lines of the snapshot (one row per player, market, book)
    lines/<date>_manual.parquet     hand-entered lines for the date (source = manual)
    usage/odds_api.jsonl            every call to The Odds API with its cost (nba.lines.guard)
    recommendations/<date>.json     the private view for a captured night

`pull` brings the whole repo into a local root (the warehouse loader reads lines/ and
captures/ from there); `push` uploads named files in one commit. Writes are refused during
a rehearsal like every other Hugging Face write.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from huggingface_hub import CommitOperationAdd, snapshot_download

from nba import config
from nba.storage import hf

PREFIXES: tuple[str, ...] = (
    config.CAPTURES_DIR.as_posix(),
    config.LINES_DIR.as_posix(),
    config.ODDS_USAGE_FILE.parent.as_posix(),
    config.RECOMMENDATIONS_DIR.as_posix(),
)


def ensure_private_repo(repo_id: str = config.HF_LINES_REPO) -> None:
    """Create the private dataset repo if it does not exist (private=True is enforced)."""
    api = hf._api(require_token=True)
    api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
    info = api.dataset_info(repo_id)
    if not info.private:
        raise RuntimeError(f"{repo_id} is not private; refusing to use it for raw lines")


def pull(root: Path, repo_id: str = config.HF_LINES_REPO, revision: str | None = None) -> str:
    api = hf._api(require_token=True)  # a private repo needs the token even to read
    sha = api.dataset_info(repo_id, revision=revision).sha
    with tempfile.TemporaryDirectory() as tmp:
        snapshot_download(
            repo_id,
            repo_type="dataset",
            revision=sha,
            allow_patterns=[f"{p}/**" for p in PREFIXES],
            local_dir=tmp,
            token=config.hf_token(),
        )
        for prefix in PREFIXES:
            src_dir = Path(tmp) / prefix
            if not src_dir.is_dir():
                continue
            for src in src_dir.rglob("*"):
                if src.is_file():
                    dest = root / src.relative_to(tmp)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(src, dest)
    return sha


def push(root: Path, files: list[Path], message: str, repo_id: str = config.HF_LINES_REPO) -> str:
    """Upload the given files (paths under `root`, inside the known prefixes) in one commit."""
    operations = []
    for path in files:
        rel = path.resolve().relative_to(root.resolve())
        if rel.parts[0] not in PREFIXES:
            raise ValueError(f"{path} is not inside a lines-repo folder {PREFIXES}")
        operations.append(
            CommitOperationAdd(path_in_repo=rel.as_posix(), path_or_fileobj=str(path))
        )
    if not operations:
        raise ValueError("nothing to push")
    api = hf._api(require_token=True)
    info = api.create_commit(
        repo_id=repo_id, repo_type="dataset", operations=operations, commit_message=message
    )
    return info.oid
