"""A thin client for The Odds API v4 (docs/AE_AUDIT.md §9.1).

Two endpoints only: `/sports/{sport}/events` (free: it does not count against the quota) and
`/sports/{sport}/events/{id}/odds` (cost = unique markets returned × region units; a
`bookmakers` filter of up to ten books is one region unit). Every response's usage headers
(`x-requests-remaining`, `x-requests-used`, `x-requests-last`) are returned with the body so
the guard can log them. Nothing here retries: a failed call is reported once and the caller
decides (ADR-0031: never in a loop).

The key is read once in nba.config (`ODDS_API_KEY`) and never logged.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import requests

from nba import config

TIMEOUT = 20  # seconds


class OddsApiError(RuntimeError):
    """A non-2xx answer; the status code and the provider's message, never the key."""


@dataclass(frozen=True)
class Usage:
    remaining: int | None
    used: int | None
    last: int | None
    at: str

    @classmethod
    def from_headers(cls, headers: Any) -> Usage:
        def _int(name: str) -> int | None:
            value = headers.get(name)
            try:
                return int(float(value)) if value is not None else None
            except ValueError:
                return None

        return cls(
            remaining=_int("x-requests-remaining"),
            used=_int("x-requests-used"),
            last=_int("x-requests-last"),
            at=datetime.now(UTC).isoformat(timespec="seconds"),
        )


@dataclass(frozen=True)
class Answer:
    data: Any
    usage: Usage
    endpoint: str


def event_odds_cost(n_markets: int, n_bookmakers: int = len(config.ODDS_BOOKMAKERS)) -> int:
    """Credits one per-event odds call costs under the documented rule
    (markets × region units; every group of ten bookmakers is one region unit)."""
    region_units = max(1, -(-n_bookmakers // 10))
    return n_markets * region_units


class OddsApi:
    def __init__(self, api_key: str | None = None, session: requests.Session | None = None):
        self.api_key = api_key or config.ODDS_API_KEY
        if not self.api_key:
            raise RuntimeError("ODDS_API_KEY is not set")
        self.session = session or requests.Session()

    def _get(self, path: str, params: dict[str, Any]) -> Answer:
        url = f"{config.ODDS_API_BASE}{path}"
        r = self.session.get(url, params={**params, "apiKey": self.api_key}, timeout=TIMEOUT)
        usage = Usage.from_headers(r.headers)
        if r.status_code >= 400:
            body = r.text[:300].replace(self.api_key, "***")
            raise OddsApiError(f"{path}: HTTP {r.status_code}: {body}")
        return Answer(data=r.json(), usage=usage, endpoint=path)

    def events(self, commence_from: str | None = None, commence_to: str | None = None) -> Answer:
        """Upcoming and live events (0 credits). ISO-8601 UTC bounds, e.g. 2026-10-20T00:00:00Z."""
        params: dict[str, Any] = {"dateFormat": "iso"}
        if commence_from:
            params["commenceTimeFrom"] = commence_from
        if commence_to:
            params["commenceTimeTo"] = commence_to
        return self._get(f"/sports/{config.ODDS_SPORT}/events", params)

    def event_odds(
        self,
        event_id: str,
        markets: tuple[str, ...] = config.ODDS_MARKETS,
        bookmakers: tuple[str, ...] = config.ODDS_BOOKMAKERS,
    ) -> Answer:
        """Player props for one event, American odds; costs event_odds_cost(len(markets))."""
        return self._get(
            f"/sports/{config.ODDS_SPORT}/events/{event_id}/odds",
            {
                "markets": ",".join(markets),
                "bookmakers": ",".join(bookmakers),
                "oddsFormat": "american",
                "dateFormat": "iso",
            },
        )
