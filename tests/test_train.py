import json
from pathlib import Path

import pandas as pd

from nba import config
from nba.features import asof
from nba.models import evaluate, publish, train

SMOKE_PARAMS = {**config.LGBM_PARAMS, "n_estimators": 20, "min_child_samples": 5}


def test_train_smoke(game_logs: pd.DataFrame, tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    results = train.run(game_logs, models_dir, params=SMOKE_PARAMS)

    assert set(results["targets"]) == set(config.TARGETS)
    for target in config.TARGETS:
        assert (models_dir / f"lgbm_{target}.txt").exists()
        per_target = results["targets"][target]
        assert set(per_target) == {"model", "baseline_last10", "baseline_season"}
        ns = {name: m["n"] for name, m in per_target.items()}
        assert len(set(ns.values())) == 1, "all predictors must be scored on the same rows"
        for m in per_target.values():
            assert set(m) == set(evaluate.METRIC_KEYS)
            assert m["mae"] >= 0 and m["rmse"] >= m["mae"]

    split = results["split"]
    assert split["holdout_season"] == config.HOLDOUT_SEASON
    assert split["train_seasons"] == list(config.TRAIN_SEASONS)
    assert split["train_dates"]["end"] < split["holdout_dates"]["start"]
    assert results["features"] == asof.FEATURE_COLUMNS


def test_split_respects_seasons_and_min_minutes(game_logs: pd.DataFrame) -> None:
    features = asof.build_features(game_logs)
    train_rows, holdout_rows = train.split(features)
    assert set(train_rows["season"]) <= set(config.TRAIN_SEASONS)
    assert set(holdout_rows["season"]) == {config.HOLDOUT_SEASON}
    assert (train_rows["minutes"] >= config.MIN_MINUTES).all()
    assert (holdout_rows["minutes"] >= config.MIN_MINUTES).all()


def test_metrics_json_and_model_card(game_logs: pd.DataFrame, tmp_path: Path) -> None:
    results = train.run(game_logs, tmp_path / "models", params=SMOKE_PARAMS)
    path = tmp_path / "reports" / "metrics.json"
    payload = evaluate.write_metrics(results, path, dataset_version="local:abc123")

    on_disk = json.loads(path.read_text())
    assert on_disk == payload
    for key in (
        "generated_at",
        "git_sha",
        "dataset",
        "split",
        "features",
        "metrics",
        "model_files",
    ):
        assert key in on_disk
    assert on_disk["dataset"] == {"repo": config.HF_DATASET_REPO, "version": "local:abc123"}
    assert on_disk["split"]["holdout_season"] == config.HOLDOUT_SEASON

    table = evaluate.metrics_table(payload)
    assert "| pts | model |" in table and "| ast | baseline_season |" in table

    card = publish.model_card(payload)
    assert card.startswith("---\nlicense: mit")
    assert "non-commercial" in card
    assert "baseline_last10" in card
    assert "lgbm_pts.txt" in card
