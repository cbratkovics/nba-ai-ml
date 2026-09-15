"""The provenance report: built from injected downloads, and the committed one is consistent."""

import json
from pathlib import Path

import pandas as pd

from nba import config
from nba.storage import provenance

ROOT = config.REPO_ROOT


def test_build_hashes_every_input(tmp_path: Path) -> None:
    files: dict[tuple[str, str], Path] = {}

    def make(repo: str, rel: str, content: bytes) -> Path:
        path = tmp_path / repo.replace("/", "_") / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        files[(repo, rel)] = path
        return path

    for season in config.SEASONS:
        pq = tmp_path / f"{season}.parquet"
        pd.DataFrame({"a": [1, 2, 3]}).to_parquet(pq, index=False)
        make(config.HF_DATASET_REPO, f"game_logs/game_logs_{season}.parquet", pq.read_bytes())
    for name in provenance.MODEL_FILES:
        make(config.HF_MODEL_REPO, name, f"model {name}".encode())
    make(config.HF_DATASET_REPO, "replay/2025-26/replay.json", b"{}")
    make(config.HF_DATASET_REPO, "replay/2025-26/daily_mae.json", b"{}")
    repo_root = tmp_path / "repo"
    (repo_root / "reports").mkdir(parents=True)
    (repo_root / "reports" / "metrics.json").write_bytes(b"model metrics.json")
    (repo_root / "reports" / "replay_2025-26.json").write_bytes(b"{}")

    lfs = {
        rel: provenance.sha256_of(path)
        for (repo, rel), path in files.items()
        if rel.endswith(".parquet")
    }
    report = provenance.build(
        download=lambda repo, rt, rel, rev: files[(repo, rel)],
        lfs_sha256=lambda repo, rt, rev: lfs,
        repo_sha=lambda repo, rt, rev: "mainsha",
        repo_root=repo_root,
        dataset_revision="d" * 40,
        model_revision="m" * 40,
        git_sha="g" * 40,
    )
    assert set(report["dataset"]["files"]) == {
        f"game_logs/game_logs_{s}.parquet" for s in config.SEASONS
    }
    first = report["dataset"]["files"][f"game_logs/game_logs_{config.SEASONS[0]}.parquet"]
    assert first["rows"] == 3 and first["matches_hub_lfs_sha256"] is True
    assert report["model"]["files"]["lgbm_pts.txt"]["sha256"] == provenance.sha256_of(
        files[(config.HF_MODEL_REPO, "lgbm_pts.txt")]
    )
    assert report["checks"] == {
        "model_repo_metrics_json_equals_committed": True,
        "hub_replay_json_equals_committed_report": True,
        "all_parquet_match_hub_lfs_sha256": True,
    }
    assert report["hub_products"]["replay/2025-26/replay.json"]["revision"] == "mainsha"
    assert report["model_identity"] == config.model_identity()
    assert json.dumps(report)


def test_committed_provenance_is_consistent_with_config_and_reports() -> None:
    path = ROOT / provenance.provenance_path()
    report = json.loads(path.read_text())
    assert report["dataset"]["revision"] == config.DATASET_REVISION
    assert report["model"]["revision"] == config.MODEL_REVISION
    assert report["model"]["commit"] == config.MODEL_COMMIT
    assert set(report["dataset"]["files"]) == {
        f"{config.HF_DATASET_PREFIX}/game_logs_{s}.parquet" for s in config.SEASONS
    }
    for rel in ("reports/metrics.json", "reports/replay_2025-26.json"):
        assert report["reports"][rel]["sha256"] == provenance.sha256_of(ROOT / rel), rel
    assert report["checks"]["all_parquet_match_hub_lfs_sha256"] is True
    assert report["checks"]["model_repo_metrics_json_equals_committed"] is True
