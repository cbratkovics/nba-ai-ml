"""pull_dataset restores the per-season Parquet layout that push_dataset uploads."""

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from nba import config
from nba.storage import hf, local


def test_pull_dataset_restores_per_season_layout(
    game_logs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # What the repo would contain after push_dataset: game_logs/game_logs_<season>.parquet.
    repo = tmp_path / "repo"
    local.write_per_season(game_logs, repo / config.HF_DATASET_PREFIX)
    (repo / "README.md").write_text("# card\n")

    seen: dict = {}

    def fake_snapshot_download(repo_id, repo_type, revision, allow_patterns, local_dir, token):
        seen.update(
            repo_id=repo_id, repo_type=repo_type, revision=revision, patterns=allow_patterns
        )
        import shutil

        for src in (repo / config.HF_DATASET_PREFIX).glob("*.parquet"):
            dst = Path(local_dir) / config.HF_DATASET_PREFIX / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
        return local_dir

    class FakeApi:
        def __init__(self, token=None):
            pass

        def dataset_info(self, repo_id, revision=None):
            return SimpleNamespace(sha="feedface")

    monkeypatch.setattr(hf, "snapshot_download", fake_snapshot_download)
    monkeypatch.setattr(hf, "HfApi", FakeApi)

    dest = tmp_path / "data" / "game_logs"
    sha = hf.pull_dataset(dest)

    assert sha == "feedface"
    assert seen["repo_id"] == config.HF_DATASET_REPO and seen["repo_type"] == "dataset"
    assert seen["revision"] == "feedface"
    assert seen["patterns"] == ["game_logs/game_logs_*.parquet"]
    assert sorted(p.name for p in local.list_parquet(dest)) == sorted(
        f"game_logs_{s}.parquet" for s in game_logs["season"].unique()
    )
    assert (dest / local.REVISION_FILE).read_text().strip() == "feedface"
    assert local.dataset_fingerprint(dest) == "hf:feedface"

    # train.yml's next step reads this directory.
    round_trip = local.read_game_logs(dest)
    pd.testing.assert_frame_equal(
        round_trip.sort_values(["game_date", "game_id", "player_id"]).reset_index(drop=True),
        game_logs.sort_values(["game_date", "game_id", "player_id"]).reset_index(drop=True),
    )


def test_pull_dataset_fails_when_repo_has_no_parquet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(hf, "snapshot_download", lambda *a, **k: k["local_dir"])

    class FakeApi:
        def __init__(self, token=None):
            pass

        def dataset_info(self, repo_id, revision=None):
            return SimpleNamespace(sha="empty")

    monkeypatch.setattr(hf, "HfApi", FakeApi)
    with pytest.raises(FileNotFoundError, match="game_logs/game_logs_"):
        hf.pull_dataset(tmp_path / "data")
