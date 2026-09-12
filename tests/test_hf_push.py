"""push_dataset builds one commit with the per-season Parquet files and the card."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from nba import config
from nba.storage import hf, local


class FakeApi:
    def __init__(self, token=None):
        self.token = token
        self.created: list[tuple[str, str]] = []
        self.commits: list[dict] = []

    def create_repo(self, repo_id, repo_type, exist_ok):
        self.created.append((repo_id, repo_type))

    def create_commit(self, repo_id, repo_type, operations, commit_message):
        self.commits.append(
            {
                "repo_id": repo_id,
                "repo_type": repo_type,
                "paths": [(op.path_in_repo, str(op.path_or_fileobj)) for op in operations],
                "message": commit_message,
            }
        )
        return SimpleNamespace(oid="abc123")


def test_push_dataset_uploads_parquet_and_card(
    game_logs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = tmp_path / "game_logs"
    written = local.write_per_season(game_logs, data_dir)
    card = data_dir / "README.md"
    card.write_text("# card\n")
    fake = FakeApi()
    monkeypatch.setattr(hf, "HfApi", lambda token=None: fake)
    monkeypatch.setattr(config, "HF_TOKEN", "test-token")

    sha = hf.push_dataset(data_dir, card_path=card, message="test push")

    assert sha == "abc123"
    assert fake.created == [(config.HF_DATASET_REPO, "dataset")]
    (commit,) = fake.commits
    assert commit["repo_id"] == config.HF_DATASET_REPO and commit["repo_type"] == "dataset"
    assert commit["message"] == "test push"
    expected = [(f"game_logs/{p.name}", str(p)) for p in written.values()]
    assert commit["paths"] == sorted(expected) + [("README.md", str(card))]


def test_push_dataset_without_card_uploads_only_parquet(
    game_logs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = tmp_path / "game_logs"
    local.write_per_season(game_logs, data_dir)
    fake = FakeApi()
    monkeypatch.setattr(hf, "HfApi", lambda token=None: fake)
    monkeypatch.setattr(config, "HF_TOKEN", "test-token")
    hf.push_dataset(data_dir)
    assert all(path.startswith("game_logs/") for path, _ in fake.commits[0]["paths"])


def test_push_requires_token(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, game_logs) -> None:
    data_dir = tmp_path / "game_logs"
    local.write_per_season(game_logs, data_dir)
    monkeypatch.setattr(config, "HF_TOKEN", None)
    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        hf.push_dataset(data_dir)
