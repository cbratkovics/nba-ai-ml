import fnmatch
import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from nba import config
from nba.predict import residuals
from nba.storage import hf


def _predictions_from_actuals(game_logs: pd.DataFrame, d: pd.Timestamp) -> pd.DataFrame:
    """Predictions for one game date, built from actual rows so residuals are known."""
    rows = game_logs[game_logs["game_date"] == d].copy()
    assert len(rows) >= 8
    preds = pd.DataFrame(
        {
            "date": d.date().isoformat(),
            "game_id": rows["game_id"].to_numpy(),
            "player_id": rows["player_id"].to_numpy(),
            "player_name": rows["player_name"].to_numpy(),
            "team": rows["team"].to_numpy(),
            "opponent": rows["opponent"].to_numpy(),
            "home": rows["home"].to_numpy(),
            "pred_pts": (rows["pts"] + 2).astype("float64").to_numpy(),
            "pred_reb": (rows["reb"] - 1).astype("float64").to_numpy(),
            "pred_ast": rows["ast"].astype("float64").to_numpy(),
            "pts_mean_last10": 10.0,
            "reb_mean_last10": 5.0,
            "ast_mean_last10": 3.0,
            "pts_mean_season": 10.0,
            "reb_mean_season": 5.0,
            "ast_mean_season": 3.0,
            "games_played_season": 5,
            "model_revision": "rev-m",
            "dataset_revision": "rev-d",
            "generated_at": "2026-01-01T00:00:00+00:00",
        }
    )
    return preds


def test_compute_reports_missing_actuals_and_maes(game_logs: pd.DataFrame, tmp_path: Path) -> None:
    dates = sorted(game_logs["game_date"].unique())
    d = pd.Timestamp(dates[5])
    preds = _predictions_from_actuals(game_logs, d)
    # One predicted player who did not play, and one predicted game that is not ingested.
    dnp = preds.iloc[[0]].copy()
    dnp["player_id"] = 424242
    not_ingested = preds.iloc[[1]].copy()
    not_ingested["game_id"] = "0029999999"
    preds = pd.concat([preds, dnp, not_ingested], ignore_index=True)
    # Remove one actual row from a predicted game so it counts as an unpredicted actual.
    logs = game_logs.copy()
    victim = preds.iloc[2]
    preds = preds.drop(index=2).reset_index(drop=True)

    result = residuals.compute(d.date(), preds, logs, tmp_path / "residuals")
    line = result.line
    n_real = len(preds) - 2
    assert line["n_predicted"] == len(preds)
    assert line["n_with_actuals"] == n_real
    assert line["n_missing_actuals"] == 2
    assert line["n_missing_game_not_ingested"] == 1
    assert line["n_missing_player_did_not_play"] == 1
    assert line["n_unpredicted_actuals"] == 1
    assert line["mae"] == {
        "pts": pytest.approx(2.0),
        "reb": pytest.approx(1.0),
        "ast": pytest.approx(0.0),
    }
    # Restricted population: minutes >= MIN_MINUTES and both baselines present.
    played = result.residuals[result.residuals["has_actual"]]
    assert line["n_restricted"] == int((played["minutes"] >= config.MIN_MINUTES).sum())
    assert line["model_revision"] == "rev-m"
    on_disk = pd.read_parquet(tmp_path / "residuals" / f"{d.date().isoformat()}.parquet")
    assert list(on_disk.columns) == list(residuals.RESIDUAL_COLUMNS)
    assert on_disk["has_actual"].sum() == n_real
    assert "2 missing (1 game not ingested, 1 did not play)" in result.message
    assert int(victim["player_id"]) not in set(on_disk["player_id"])


def test_rolling_line_is_appended_replaced_and_windowed(
    game_logs: pd.DataFrame, tmp_path: Path
) -> None:
    dates = sorted(game_logs["game_date"].unique())
    pdir, rdir = tmp_path / "predictions", tmp_path / "residuals"
    d1, d2 = pd.Timestamp(dates[3]), pd.Timestamp(dates[4])
    r1 = residuals.compute(d1.date(), _predictions_from_actuals(game_logs, d1), game_logs, rdir)
    residuals.append_rolling_line(pdir, r1.line)
    r2 = residuals.compute(d2.date(), _predictions_from_actuals(game_logs, d2), game_logs, rdir)
    path = residuals.append_rolling_line(pdir, r2.line)
    lines = [json.loads(ln) for ln in path.read_text().splitlines()]
    assert [ln["date"] for ln in lines] == [d1.date().isoformat(), d2.date().isoformat()]
    assert lines[1]["rolling_30d"]["days"] == 2
    assert lines[1]["rolling_30d"]["n"] == lines[0]["n_with_actuals"] + lines[1]["n_with_actuals"]
    assert lines[1]["rolling_30d"]["mae"]["pts"] == pytest.approx(2.0)
    # Re-running the same date replaces its line instead of duplicating it.
    residuals.append_rolling_line(pdir, r2.line)
    assert len(path.read_text().splitlines()) == 2
    # A far-future date sees no files in its 30-day window.
    far = residuals.rolling_mae(rdir, date(2030, 1, 1))
    assert far == {"days": 0, "n": 0, "mae": {"pts": None, "reb": None, "ast": None}}


def test_cli_without_predictions_file_is_a_clean_noop(
    game_logs: pd.DataFrame, tmp_path: Path, capsys
) -> None:
    from nba.storage import local

    data_dir = tmp_path / "game_logs"
    local.write_per_season(game_logs, data_dir)
    rc = residuals.main(
        [
            "--date",
            "2026-09-11",
            "--data-dir",
            str(data_dir),
            "--predictions-dir",
            str(tmp_path / "p"),
            "--residuals-dir",
            str(tmp_path / "r"),
            "--no-pull",
        ]
    )
    assert rc == 0
    assert "RESIDUALS 2026-09-11: no predictions file" in capsys.readouterr().out


def test_push_and_pull_products(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "root"
    (root / "predictions").mkdir(parents=True)
    (root / "residuals").mkdir()
    (root / "predictions" / "2026-01-15.parquet").write_bytes(b"p")
    (root / "predictions" / "latest.json").write_text("{}")
    (root / "residuals" / "2026-01-14.parquet").write_bytes(b"r")
    commits = []

    class FakeApi:
        def __init__(self, token=None):
            pass

        def create_commit(self, repo_id, repo_type, operations, commit_message):
            commits.append(sorted(op.path_in_repo for op in operations))
            return SimpleNamespace(oid="prodsha")

        def dataset_info(self, repo_id, revision=None):
            return SimpleNamespace(sha="prodsha")

    monkeypatch.setattr(hf, "HfApi", FakeApi)
    monkeypatch.setattr(config, "HF_TOKEN", "t")
    assert hf.push_products(root) == "prodsha"
    assert commits == [
        [
            "predictions/2026-01-15.parquet",
            "predictions/latest.json",
            "residuals/2026-01-14.parquet",
        ]
    ]
    assert hf.push_products(tmp_path / "empty") is None

    def fake_snapshot(
        repo_id, repo_type, revision, allow_patterns, ignore_patterns, local_dir, token
    ):
        assert allow_patterns == list(hf.PRODUCT_PATTERNS)
        # The per-date replay slates are excluded from every pull; the rest of replay/ is not.
        assert ignore_patterns == ["replay/*/slates/*"]
        assert fnmatch.fnmatch("replay/2025-26/slates/2026-04-12.json", ignore_patterns[0])
        assert fnmatch.fnmatch("replay/2025-26/slates/index.json", ignore_patterns[0])
        assert not fnmatch.fnmatch("replay/2025-26/daily_mae.json", ignore_patterns[0])
        residual = "replay/2025-26/residuals/2026-04-12.parquet"
        assert not fnmatch.fnmatch(residual, ignore_patterns[0])
        import shutil

        for prefix in ("predictions", "residuals"):
            shutil.copytree(root / prefix, Path(local_dir) / prefix)
        return local_dir

    monkeypatch.setattr(hf, "snapshot_download", fake_snapshot)
    dest = tmp_path / "dest"
    assert hf.pull_products(dest) == "prodsha"
    assert sorted(p.name for p in (dest / "predictions").iterdir()) == [
        "2026-01-15.parquet",
        "latest.json",
    ]
    assert (dest / "residuals" / "2026-01-14.parquet").read_bytes() == b"r"
