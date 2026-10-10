"""The hand-entered CSV is validated on load and comes out in the lines layout, source manual."""

from __future__ import annotations

from pathlib import Path

import pytest

from nba.lines import manual

GOOD = (
    manual.template_text()
    + "2026-10-20T20:05:00Z,2026-10-20,caesars,Jayson Tatum,BOS,player_points,27.5,-115,-105,\n"
    + "2026-10-20T20:06:00Z,2026-10-20,draftkings,Jaylen Brown,BOS,player_rebounds,6.5,"
    + "-110,-110,note\n"
)


def test_good_file_loads(tmp_path: Path) -> None:
    p = tmp_path / "lines.csv"
    p.write_text(GOOD)
    got = manual.load(p)
    assert list(got["source"]) == ["manual", "manual"]
    assert got["line"].tolist() == [27.5, 6.5] and got["over_price"].tolist() == [-115, -110]
    assert str(got["read_at_utc"].iloc[0]) == "2026-10-20 20:05:00+00:00"
    assert got["book"].tolist() == ["caesars", "draftkings"]


@pytest.mark.parametrize(
    "bad_row,needle",
    [
        ("20/10/2026 20:05,2026-10-20,caesars,A B,BOS,player_points,27.5,-115,-105,", "ISO-8601"),
        (
            "2026-10-20T20:05:00,2026-10-20,caesars,A B,BOS,player_points,27.5,-115,-105,",
            "timezone",
        ),
        ("2026-10-20T20:05:00Z,2026-10-20,bet365,A B,BOS,player_points,27.5,-115,-105,", "book"),
        ("2026-10-20T20:05:00Z,2026-10-20,caesars,A B,BOS,player_threes,2.5,-115,-105,", "market"),
        ("2026-10-20T20:05:00Z,2026-10-20,caesars,A B,BOS,player_points,27.3,-115,-105,", "half"),
        (
            "2026-10-20T20:05:00Z,2026-10-20,caesars,A B,BOS,player_points,27.5,-90,-105,",
            "American",
        ),
        ("2026-10-20T20:05:00Z,2026-10-20,caesars,,BOS,player_points,27.5,-115,-105,", "required"),
    ],
)
def test_each_rule_is_enforced(tmp_path: Path, bad_row: str, needle: str) -> None:
    p = tmp_path / "lines.csv"
    p.write_text(manual.template_text() + bad_row + "\n")
    with pytest.raises(manual.ManualLinesError) as info:
        manual.load(p)
    assert any(needle in problem for problem in info.value.problems), info.value.problems


def test_duplicates_and_missing_columns(tmp_path: Path) -> None:
    p = tmp_path / "lines.csv"
    row = "2026-10-20T20:05:00Z,2026-10-20,caesars,A B,BOS,player_points,27.5,-115,-105,\n"
    p.write_text(manual.template_text() + row + row)
    with pytest.raises(manual.ManualLinesError, match="duplicate"):
        manual.load(p)
    p.write_text("read_at_utc,game_date\n")
    with pytest.raises(manual.ManualLinesError, match="missing columns"):
        manual.load(p)
