"""The private lines repo: every push checks the repo is private first (ADR-0030 amendment 5)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from nba import config
from nba.lines import store
from nba.storage import hf


class FakeApi:
    def __init__(self, private: bool):
        self._private = private
        self.commits: list[list[str]] = []
        self.created: list[dict] = []

    def dataset_info(self, repo_id, revision=None):
        return SimpleNamespace(private=self._private, sha="sha")

    def create_commit(self, repo_id, repo_type, operations, commit_message):
        self.commits.append([op.path_in_repo for op in operations])
        return SimpleNamespace(oid="newsha")

    def create_repo(self, repo_id, repo_type, private, exist_ok):
        self.created.append({"repo_id": repo_id, "private": private})


def _files(root: Path) -> list[Path]:
    p = root / config.LINES_DIR / "2026-10-20.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x")
    return [p]


def test_push_refuses_a_public_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    api = FakeApi(private=False)
    monkeypatch.setattr(hf, "_api", lambda require_token: api)
    with pytest.raises(RuntimeError, match="not private"):
        store.push(tmp_path, _files(tmp_path), "m")
    assert api.commits == []


def test_push_to_a_private_repo_commits_inside_known_prefixes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = FakeApi(private=True)
    monkeypatch.setattr(hf, "_api", lambda require_token: api)
    assert store.push(tmp_path, _files(tmp_path), "m") == "newsha"
    assert api.commits == [["lines/2026-10-20.parquet"]]
    stray = tmp_path / "elsewhere" / "x.parquet"
    stray.parent.mkdir()
    stray.write_bytes(b"x")
    with pytest.raises(ValueError, match="not inside"):
        store.push(tmp_path, [stray], "m")


def test_ensure_private_repo_creates_private_and_refuses_public(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = FakeApi(private=True)
    monkeypatch.setattr(hf, "_api", lambda require_token: api)
    store.ensure_private_repo()
    assert api.created == [{"repo_id": config.HF_LINES_REPO, "private": True}]
    monkeypatch.setattr(hf, "_api", lambda require_token: FakeApi(private=False))
    with pytest.raises(RuntimeError, match="not private"):
        store.ensure_private_repo()
