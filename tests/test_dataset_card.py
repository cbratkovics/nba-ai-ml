# ruff: noqa: E501  (the placeholder card keeps the upstream README's long lines verbatim)
import pandas as pd
import pytest

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
    assert "| 2021-22 | 25,826 | 2021-10-19 | 2022-04-10 |" in out
    assert "| 2025-26 | 26,648 | 2025-10-21 | 2026-04-12 |" in out
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
        "| Season | Rows | First game | Last game |",
        "|---|---|---|---|",
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
