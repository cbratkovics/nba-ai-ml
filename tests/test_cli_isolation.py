"""The backfill and HF push must work on a machine where LightGBM cannot be imported."""

import subprocess
import sys
from pathlib import Path

import pytest

from nba.ingest import kaggle_backfill
from nba.storage import hf
from tests.test_dataset_card import PLACEHOLDER_CARD


def test_ingest_and_storage_do_not_import_models() -> None:
    code = (
        "import sys\n"
        "sys.modules['lightgbm'] = None\n"  # makes `import lightgbm` raise ImportError
        "import nba.ingest.kaggle_backfill, nba.storage.hf, nba.storage.local\n"
        "loaded = sorted(m for m in sys.modules if m.startswith('nba.models'))\n"
        "assert not loaded, loaded\n"
        "print('ok')\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_backfill_push_flag_calls_push_dataset(
    kaggle_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out_dir = tmp_path / "out"
    calls: list[tuple[Path, Path | None]] = []

    def fake_push(data_dir: Path, card_path: Path | None = None) -> str:
        calls.append((data_dir, card_path))
        return "deadbeef"

    monkeypatch.setattr(hf, "push_dataset", fake_push)
    monkeypatch.setattr(hf, "fetch_dataset_card", lambda: PLACEHOLDER_CARD)
    kaggle_backfill.main(["--kaggle-dir", str(kaggle_dir), "--out-dir", str(out_dir), "--push"])

    assert calls == [(out_dir, out_dir / "README.md")]
    card = (out_dir / "README.md").read_text()
    assert "[n]" not in card and "| 2024-25 |" in card
    assert "deadbeef" in capsys.readouterr().out


def test_backfill_without_push_flag_does_not_push(
    kaggle_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_push(data_dir: Path, card_path: Path | None = None) -> str:
        raise AssertionError("push_dataset must not be called without --push")

    monkeypatch.setattr(hf, "push_dataset", fail_push)
    kaggle_backfill.main(["--kaggle-dir", str(kaggle_dir), "--out-dir", str(tmp_path / "out")])
