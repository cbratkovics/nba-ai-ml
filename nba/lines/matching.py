"""Sportsbook player names to player_id, deterministically (ADR-0031).

Three steps, no fuzzy matching anywhere:
  1. normalise both sides (lower case, accents stripped, periods and apostrophes removed,
     generational suffixes Jr/Sr/II/III/IV dropped, whitespace collapsed);
  2. exact match on the normalised name within the player's team roster (the players who
     appeared for that team this season or last), then within every player seen in those
     seasons when the team match fails and the name is unique league-wide;
  3. the committed alias file (nba/lines/aliases.csv: raw sportsbook name -> player_id, with a
     note saying how it was confirmed) for the names the normaliser cannot bridge.

Anything else is `unmatched`: counted, published in the capture record, never guessed.
"""

from __future__ import annotations

import csv
import re
import unicodedata
from pathlib import Path

import pandas as pd

from nba import config

ALIASES_PATH = config.REPO_ROOT / "nba" / "lines" / "aliases.csv"
SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}
METHODS = ("team", "league", "alias", "unmatched")


def normalise(name: str) -> str:
    text = unicodedata.normalize("NFKD", str(name))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().replace(".", "").replace("'", "").replace("’", "")
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    tokens = [t for t in text.split() if t not in SUFFIXES]
    return " ".join(tokens)


def load_aliases(path: Path = ALIASES_PATH) -> dict[str, int]:
    if not path.exists():
        return {}
    with path.open(newline="") as fh:
        return {normalise(r["raw_name"]): int(r["player_id"]) for r in csv.DictReader(fh)}


def roster_index(game_logs: pd.DataFrame, seasons: tuple[str, ...]) -> pd.DataFrame:
    """One row per (player_id, team) seen in the given seasons, with the normalised name."""
    logs = game_logs[game_logs["season"].isin(seasons)]
    latest = logs.sort_values("game_date").groupby("player_id")["player_name"].last()
    teams = logs[["player_id", "team"]].drop_duplicates()
    out = teams.merge(latest.rename("player_name").reset_index(), on="player_id")
    out["norm"] = out["player_name"].map(normalise)
    return out


def match_players(
    names: pd.DataFrame, roster: pd.DataFrame, aliases: dict[str, int] | None = None
) -> pd.DataFrame:
    """`names` has raw_name and team (abbreviation, or null). Returns one row per input row with
    player_id (nullable Int64), method and n_candidates; the input order is preserved."""
    aliases = load_aliases() if aliases is None else aliases
    by_team = roster.groupby(["team", "norm"])["player_id"].agg(lambda s: sorted(set(s)))
    league = roster.groupby("norm")["player_id"].agg(lambda s: sorted(set(s)))
    out = []
    for raw, team in zip(names["raw_name"], names["team"], strict=True):
        norm = normalise(raw)
        pid, method, n = None, "unmatched", 0
        if team is not None and (team, norm) in by_team.index:
            cands = by_team[(team, norm)]
            n = len(cands)
            if n == 1:
                pid, method = cands[0], "team"
        if pid is None and norm in league.index:
            cands = league[norm]
            n = len(cands)
            if n == 1:
                pid, method = cands[0], "league"
        if pid is None and norm in aliases:
            pid, method, n = aliases[norm], "alias", 1
        out.append(
            {"raw_name": raw, "team": team, "player_id": pid, "method": method, "n_candidates": n}
        )
    df = pd.DataFrame(out, columns=["raw_name", "team", "player_id", "method", "n_candidates"])
    df["player_id"] = df["player_id"].astype("Int64")
    return df
