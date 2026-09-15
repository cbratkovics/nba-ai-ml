"""One model identity everywhere: config, the committed reports, and the site's data layer."""

import json
import re

from nba import config

ROOT = config.REPO_ROOT


def _ts_const(name: str) -> str:
    text = (ROOT / "frontend" / "lib" / "data.ts").read_text()
    m = re.search(rf"export const {name} = '([^']+)'", text)
    assert m, f"{name} not found in frontend/lib/data.ts"
    return m.group(1)


def test_metrics_json_identity_matches_config() -> None:
    metrics = json.loads((ROOT / "reports" / "metrics.json").read_text())
    assert metrics["git_sha"] == config.MODEL_COMMIT
    assert metrics["dataset"]["version"] == f"hf:{config.DATASET_REVISION}"
    assert metrics["dataset"]["repo"] == config.HF_DATASET_REPO


def test_replay_report_identity_matches_config() -> None:
    replay = json.loads((ROOT / "reports" / "replay_2025-26.json").read_text())
    assert replay["model_revision"] == config.MODEL_REVISION
    assert replay["dataset_revision"] == config.DATASET_REVISION
    assert replay["reference"]["metrics_json_git_sha"] == config.MODEL_COMMIT


def test_frontend_mirrors_config() -> None:
    assert _ts_const("MODEL_REVISION") == config.MODEL_REVISION
    assert _ts_const("MODEL_COMMIT") == config.MODEL_COMMIT
    assert _ts_const("DATASET_REPO") == config.HF_DATASET_REPO
    assert _ts_const("MODEL_REPO") == config.HF_MODEL_REPO
    assert _ts_const("REPLAY_SEASON") == config.HOLDOUT_SEASON


def test_identity_string() -> None:
    assert config.model_identity() == "commit 50a3b2e / HF fb427de"
    readme = (ROOT / "README.md").read_text()
    assert config.model_identity() in readme
