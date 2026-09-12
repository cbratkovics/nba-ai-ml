from pathlib import Path

import pandas as pd
import pytest

from nba import config
from nba.features import asof
from nba.models import train
from nba.predict import model
from tests.test_train import SMOKE_PARAMS


def test_load_from_dir_and_predict_round_trip(game_logs: pd.DataFrame, tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    train.run(game_logs, models_dir, params=SMOKE_PARAMS)
    models = model.load_from_dir(models_dir, revision="local-test")
    assert models.revision == "local-test" and set(models.boosters) == set(config.TARGETS)
    features = asof.build_features(game_logs).dropna(subset=["pts_mean_last10"]).head(20)
    preds = model.predict(models, features)
    assert list(preds.columns) == [f"pred_{t}" for t in config.TARGETS]
    assert len(preds) == 20 and preds.notna().all().all()


def test_missing_model_file_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="lgbm_pts.txt"):
        model.load_from_dir(tmp_path)


def test_feature_name_mismatch_is_an_error() -> None:
    class Wrong:
        def feature_name(self):
            return ["not", "the", "features"]

        def predict(self, X):
            return X.iloc[:, 0]

    with pytest.raises(ValueError, match="do not match asof.FEATURE_COLUMNS"):
        model.check_feature_names(model.Models("x", {"pts": Wrong()}))


def test_metrics_report_matches_pinned_model() -> None:
    import json

    m = json.loads(Path("reports/metrics.json").read_text())
    assert m["features"] == asof.FEATURE_COLUMNS
    assert m["model_files"] == {t: model.model_file(t) for t in config.TARGETS}
    assert m["split"]["holdout_season"] == config.HOLDOUT_SEASON
