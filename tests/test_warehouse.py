"""The warehouse loader's manifest, the nightly build gate, and the gold publisher."""

import json
from pathlib import Path

import pandas as pd

from nba import config
from nba.storage import local
from nba.warehouse import gate, load, publish


def test_write_manifest_counts_families(tmp_path: Path, game_logs: pd.DataFrame) -> None:
    root = tmp_path / "wh"
    local.write_per_season(game_logs, root / config.HF_DATASET_PREFIX)
    (root / "daily_reports").mkdir()
    (root / "daily_reports" / "2026-01-01.json").write_text("{}")
    manifest = load.write_manifest(root, "abc123", "2025-26")
    assert manifest["dataset_revision"] == "abc123"
    assert manifest["families"]["game_logs"] == game_logs["season"].nunique()
    assert manifest["families"]["daily_reports"] == 1
    assert manifest["families"]["predictions"] == 0
    assert manifest["model_revision"] == config.MODEL_REVISION
    assert json.loads((root / load.MANIFEST_FILE).read_text()) == manifest


def test_gate_skips_off_season_zero_row_runs() -> None:
    idle = {
        "ingest": {"counts": {"new": 0, "changed": 0, "unchanged": 1234}},
        "residuals": {"status": "no_predictions"},
        "slate": {"status": "no_games"},
    }
    assert gate.decide(idle) == (
        False,
        "no new rows, no residuals, no slate (off-season zero-row run)",
    )
    assert gate.decide(idle, force=True) == (True, "forced")
    assert gate.decide({**idle, "ingest": {"counts": {"new": 3, "changed": 1}}})[0]
    assert gate.decide({**idle, "residuals": {"n_with_actuals": 200}})[0]
    assert gate.decide({**idle, "slate": {"status": "ok"}}) == (True, "slate written")
    assert gate.decide({}) == (
        False,
        "no new rows, no residuals, no slate (off-season zero-row run)",
    )


def test_gate_cli_prints_one_line(tmp_path: Path, capsys) -> None:
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"ingest": {"counts": {"new": 0, "changed": 0}}}))
    assert gate.main(["--summary", str(p)]) == 0
    assert capsys.readouterr().out.startswith("WAREHOUSE skip:")
    assert gate.main(["--summary", str(tmp_path / "missing.json")]) == 0
    assert capsys.readouterr().out.startswith("WAREHOUSE build:")


def test_publish_dry_run_lists_marts_and_manifest(tmp_path: Path, capsys, monkeypatch) -> None:
    export = tmp_path / "export"
    export.mkdir()
    (export / "fct_prediction.parquet").write_bytes(b"x")
    (export / "_export_manifest.json").write_text("[]")
    assert publish.publish(export, dry_run=True) is None
    out = capsys.readouterr().out
    assert (
        f"{config.HF_GOLD_PREFIX}/fct_prediction.parquet" in out and "_export_manifest.json" in out
    )
    pushed = []
    monkeypatch.setattr("nba.storage.hf.push_gold", lambda files: pushed.append(files) or "sha1")
    assert publish.publish(export) == "sha1"
    assert [p.name for p in pushed[0]] == ["fct_prediction.parquet", "_export_manifest.json"]
