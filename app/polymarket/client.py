from typing import Any
import asyncio

import httpx

from app.models import RawMarket
from app.polymarket.normalizer import dedupe_markets, normalize_market


GAMMA_BASE_URL = "https://gamma-api.polymarket.com"
CLOB_BASE_URL = "https://clob.polymarket.com"


class PolymarketClient:
    def __init__(self, http_client: httpx.AsyncClient | None = None) -> None:
        self.http_client = http_client or httpx.AsyncClient(timeout=30.0)

    async def search_markets(self, query: str, limit: int = 75) -> list[RawMarket]:
        public_markets = await self.public_search_markets(query, limit=limit)
        if public_markets:
            return public_markets
        response = await self._get(
            f"{GAMMA_BASE_URL}/markets",
            params={"search": query, "limit": limit, "active": "true"},
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list):
            return []
        markets: list[RawMarket] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            try:
                markets.append(normalize_market(item))
            except ValueError:
                continue
        return filter_relevant(query, dedupe_markets(markets))

    async def public_search_markets(self, query: str, limit: int = 75) -> list[RawMarket]:
        response = await self._get(
            f"{GAMMA_BASE_URL}/public-search",
            params={"q": query},
        )
        response.raise_for_status()
        payload = response.json()
        events = payload.get("events", []) if isinstance(payload, dict) else []
        markets: list[RawMarket] = []
        for event in events:
            if not isinstance(event, dict):
                continue
            event_fields = {
                "id": event.get("id"),
                "title": event.get("title"),
                "slug": event.get("slug"),
            }
            for item in event.get("markets", []):
                if not isinstance(item, dict):
                    continue
                enriched = dict(item)
                enriched["events"] = [event_fields]
                try:
                    markets.append(normalize_market(enriched))
                except ValueError:
                    continue
                if len(markets) >= limit:
                    break
        return filter_relevant(query, dedupe_markets(markets))

    async def search_many(self, queries: list[str], limit_per_query: int = 75) -> list[RawMarket]:
        collected: list[RawMarket] = []
        for query in queries:
            collected.extend(await self.search_markets(query, limit=limit_per_query))
        return dedupe_markets(collected)

    async def get_midpoints(self, token_ids: list[str]) -> dict[str, float]:
        if not token_ids:
            return {}
        response = await self._get(
            f"{CLOB_BASE_URL}/midpoints",
            params={"token_ids": ",".join(token_ids)},
        )
        response.raise_for_status()
        payload: dict[str, Any] = response.json()
        return {token_id: float(value) for token_id, value in payload.items()}

    async def _get(self, url: str, params: dict[str, Any]) -> httpx.Response:
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                response = await self.http_client.get(url, params=params)
                response.raise_for_status()
                return response
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError) as error:
                last_error = error
                await asyncio.sleep(0.5 * (attempt + 1))
        if last_error is not None:
            raise last_error
        raise RuntimeError("unreachable retry state")


def filter_relevant(query: str, markets: list[RawMarket]) -> list[RawMarket]:
    required = [
        token
        for token in query.lower().replace("-", " ").split()
        if len(token) > 3 and token not in {"market", "valuation", "public"}
    ]
    if not required:
        return markets
    filtered: list[RawMarket] = []
    for market in markets:
        text = " ".join(
            [
                market.question,
                market.event_title or "",
                market.slug or "",
                str(market.raw_json.get("description") or ""),
            ]
        ).lower()
        if all(token in text for token in required):
            filtered.append(market)
    return filtered
