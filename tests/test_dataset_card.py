# ruff: noqa: E501  (the placeholder card keeps the upstream README's long lines verbatim)
import pandas as pd
import pytest

from nba import config
from nba.storage import dataset_card

# Abridged copy of the hand-written card in the HF dataset repo, with its placeholders.
PLACEHOLDER_CARD = """---
license: cc0-1.0
pretty_name: NBA Player Game Logs (2021-22 to 2025-26)
---

# NBA Player Game Logs

One row per player per regular-season game.

## Files

`game_logs/season=YYYY-YY.parquet`, one per season.

| Season | Rows | First game | Last game |
|---|---|---|---|
| 2021-22 | [n] | [date] | [date] |
| 2022-23 | [n] | [date] | [date] |
| 2023-24 | [n] | [date] | [date] |
| 2024-25 | [n] | [date] | [date] |
| 2025-26 | [n] | [date] | [date] |

## Known limitations

- Playoffs, play-in, and preseason games are excluded.
- Rows for players who did not play (DNP) are [kept with 0 minutes / dropped — confirm from backfill].
- Team abbreviations follow `TeamHistories.csv` for the season in which the game was played.
"""


def _summary() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "rows": [25826, 26648],
            "games": [1230, 1230],
            "first_game": pd.to_datetime(["2021-10-19", "2025-10-21"]),
            "last_game": pd.to_datetime(["2022-04-10", "2026-04-12"]),
            "dnp_dropped": [5495, 5611],
            "team_by_name": [25805, 0],
        },
        index=pd.Index(["2021-22", "2025-26"], name="season"),
    )


def test_render_fills_table_dnp_and_files_only() -> None:
    out = dataset_card.render_dataset_card(PLACEHOLDER_CARD, _summary())
    assert "| Season | Rows | Games | First game | Last game |" in out
    assert "| Season | Rows | First game | Last game |" not in out
    assert "| 2021-22 | 25,826 | 1,230 | 2021-10-19 | 2022-04-10 |" in out
    assert "| 2025-26 | 26,648 | 1,230 | 2025-10-21 | 2026-04-12 |" in out
    assert "Games missing from the dump" not in out  # no missing games given
    assert "[n]" not in out and "[date]" not in out
    assert "| 2022-23 |" not in out  # only seasons in the summary remain
    assert (
        "- Rows for players who did not play (DNP) are dropped: 11,106 rows "
        "(2021-22: 5,495; 2025-26: 5,611)."
    ) in out
    assert "`game_logs/game_logs_YYYY-YY.parquet`, one per season." in out
    assert dataset_card.EXCLUDED_TEXT in out
    assert "- Playoffs, play-in, and preseason games are excluded." not in out
    # Everything else is untouched.
    for line in (
        "license: cc0-1.0",
        "# NBA Player Game Logs",
        "- Team abbreviations follow `TeamHistories.csv` for the season in which the game was played.",
    ):
        assert line in out
    assert out.endswith("\n")


def test_render_is_idempotent() -> None:
    once = dataset_card.render_dataset_card(PLACEHOLDER_CARD, _summary())
    assert dataset_card.render_dataset_card(once, _summary()) == once


def test_render_requires_expected_anchors() -> None:
    with pytest.raises(ValueError, match="table header"):
        dataset_card.render_dataset_card("# no table here\n", _summary())
    no_dnp = PLACEHOLDER_CARD.replace("- Rows for players who did not play (DNP) are", "- DNP:")
    with pytest.raises(ValueError, match="DNP bullet"):
        dataset_card.render_dataset_card(no_dnp, _summary())


MISSING = (
    {
        "season": "2024-25",
        "game_id": "0022400524",
        "scheduled": "2025-01-09",
        "home": "LAL",
        "away": "CHA",
    },
    {
        "season": "2024-25",
        "game_id": "0022400988",
        "scheduled": "2025-03-17",
        "home": "SAN",
        "away": "ORL",
    },
)


def test_render_adds_missing_games_bullet_after_dnp_and_keeps_true_game_count() -> None:
    summary = _summary()
    summary.loc["2025-26", "games"] = 1223  # must not be rounded up
    out = dataset_card.render_dataset_card(PLACEHOLDER_CARD, summary, missing_games=MISSING)
    assert "| 2025-26 | 26,648 | 1,223 | 2025-10-21 | 2026-04-12 |" in out
    lines = out.split("\n")
    dnp_at = next(
        i for i, ln in enumerate(lines) if ln.startswith("- Rows for players who did not play")
    )
    bullet = lines[dnp_at + 1]
    assert bullet.startswith("- Games missing from the dump: 2 regular-season games were postponed")
    assert "never re-captured upstream" in bullet
    assert "not filled in from any other source" in bullet and "nba_api" not in bullet
    assert (
        "2024-25: 0022400524 (2025-01-09, LAL vs CHA), 0022400988 (2025-03-17, SAN vs ORL)."
        in bullet
    )
    # Idempotent: rendering the rendered card replaces, not duplicates, the bullet.
    again = dataset_card.render_dataset_card(out, summary, missing_games=MISSING)
    assert again == out
    assert again.count("Games missing from the dump") == 1


def test_render_inserts_provenance_bullets_from_config() -> None:
    out = dataset_card.render_dataset_card(PLACEHOLDER_CARD, _summary())
    lines = out.split("\n")
    heading = lines.index(dataset_card.PROVENANCE_HEADING)
    backfill, daily = dataset_card.provenance_lines()
    assert lines[heading + 2] == backfill and lines[heading + 3] == daily
    assert "nba_api" not in out
    assert config.KAGGLE_DATASET in backfill and config.KAGGLE_SOURCE in backfill
    assert config.KAGGLE_DAILY_SOURCE in daily and str(config.DAILY_LOOKBACK_DAYS) in daily


def test_render_replaces_existing_provenance_bullets() -> None:
    card = PLACEHOLDER_CARD.replace(
        "## Files",
        "## Provenance\n\n- **Historical backfill:** old text.\n"
        "- **Daily updates (from the 2026-27 season):** `nba_api` against stats.nba.com.\n\n## Files",
    )
    out = dataset_card.render_dataset_card(card, _summary())
    assert "old text" not in out and "nba_api" not in out
    assert out.count("**Historical backfill:**") == 1 and out.count("**Daily updates:**") == 1
    assert dataset_card.render_dataset_card(out, _summary()) == out
