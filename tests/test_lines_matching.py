"""Name matching is deterministic: normalised exact matches within the team, then league-wide
when unique, then the alias file; everything else is unmatched and counted (ADR-0031)."""

from __future__ import annotations

import pandas as pd

from nba.lines import matching, teams


def test_normalise_handles_accents_suffixes_and_punctuation() -> None:
    assert matching.normalise("Luka Dončić") == "luka doncic"
    assert matching.normalise("Michael Porter Jr.") == "michael porter"
    assert matching.normalise("D.J. Augustin") == "dj augustin"
    assert matching.normalise("Royce O'Neale") == "royce oneale"
    assert matching.normalise("Harry Giles III") == "harry giles"
    assert matching.normalise("  Jaren   Jackson Jr ") == "jaren jackson"


def _roster() -> pd.DataFrame:
    logs = pd.DataFrame(
        {
            "player_id": [1, 1, 2, 3, 4, 5],
            "player_name": [
                "Jayson Tatum",
                "Jayson Tatum",
                "Jaylen Brown",
                "Jalen Brown",
                "Michael Porter Jr.",
                "Jaylen Brown",
            ],
            "team": ["BOS", "BOS", "BOS", "HOU", "DEN", "MIA"],
            "season": ["2025-26"] * 6,
            "game_date": pd.to_datetime(["2026-01-01"] * 6),
        }
    )
    return matching.roster_index(logs, ("2025-26",))


def test_team_then_league_then_alias_then_unmatched() -> None:
    roster = _roster()
    names = pd.DataFrame(
        {
            "raw_name": [
                "Jayson Tatum",
                "Jaylen Brown",
                "Jaylen Brown",
                "Michael Porter",
                "Nobody Known",
                "Jalen Brown",
            ],
            "team": ["BOS", "BOS", None, None, "BOS", "DAL"],
        }
    )
    got = matching.match_players(names, roster, aliases={matching.normalise("Nobody Known"): 99})
    assert got["method"].tolist() == ["team", "team", "unmatched", "league", "alias", "league"]
    assert got["player_id"].tolist()[0:2] == [1, 2]
    assert (
        pd.isna(got["player_id"].iloc[2]) and got["n_candidates"].iloc[2] == 2
    )  # two Jaylen Browns league-wide
    assert got["player_id"].iloc[3] == 4 and got["player_id"].iloc[4] == 99
    assert got["player_id"].iloc[5] == 3  # no DAL roster hit, unique league-wide


def test_no_fuzzy_matching() -> None:
    roster = _roster()
    names = pd.DataFrame({"raw_name": ["Jason Tatum", "J. Tatum"], "team": ["BOS", "BOS"]})
    got = matching.match_players(names, roster, aliases={})
    assert got["method"].tolist() == ["unmatched", "unmatched"]


def test_alias_file_loads_and_is_header_only_today() -> None:
    assert matching.load_aliases() == {}
    text = matching.ALIASES_PATH.read_text().splitlines()[0]
    assert text == "raw_name,player_id,confirmed_on,note"


def test_team_names_cover_thirty_franchises_and_refuse_unknowns() -> None:
    assert len(set(teams.TEAM_ABBREVIATIONS.values())) == 30
    assert teams.abbreviation("San Antonio Spurs") == "SAN"
    assert teams.abbreviation("LA Clippers") == teams.abbreviation("Los Angeles Clippers") == "LAC"
    try:
        teams.abbreviation("Seattle SuperSonics")
    except teams.UnknownTeamError as exc:
        assert "add it" in str(exc)
    else:
        raise AssertionError("unknown team must raise")
