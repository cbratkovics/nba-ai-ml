"""The model card regenerates identically from reports/metrics.json and nba.config."""

from pathlib import Path

from nba import config
from nba.models import evaluate, publish


def test_committed_model_card_matches_render() -> None:
    payload = evaluate.read_metrics(config.REPO_ROOT / "reports" / "metrics.json")
    rendered = publish.model_card(payload)
    committed = (config.REPO_ROOT / publish.CARD_PATH).read_text()
    assert committed == rendered, "run: python -m nba.models.publish --card-only"


def test_model_card_names_the_kaggle_source_and_the_identity() -> None:
    payload = evaluate.read_metrics(config.REPO_ROOT / "reports" / "metrics.json")
    card = publish.model_card(payload)
    lines = config.source_lines()
    assert lines["backfill"] in card and lines["daily"] in card
    assert "nba_api" not in card
    assert config.model_identity() in card


def test_render_card_writes_the_file(tmp_path: Path) -> None:
    out = publish.render_card(config.REPO_ROOT / "reports" / "metrics.json", tmp_path / "c.md")
    assert out.read_text().startswith("---\nlicense: mit")
