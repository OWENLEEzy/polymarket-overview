from app.models import RawMarket
from app.polymarket.client import PolymarketClient


class FakeClient(PolymarketClient):
    async def search_markets(self, query: str, limit: int = 75) -> list[RawMarket]:
        if query == "Anthropic IPO":
            return [
                RawMarket(
                    market_id="m1",
                    question="Will Anthropic IPO in 2026?",
                    event_title="Anthropic IPO",
                    outcomes=["Yes", "No"],
                    outcome_prices=[0.4, 0.6],
                    raw_json={"id": "m1"},
                )
            ]
        if query == "Anthropic market cap":
            return [
                RawMarket(
                    market_id="m2",
                    question="Will Anthropic closing market cap exceed $100B?",
                    event_title="Anthropic closing market cap",
                    outcomes=["Yes", "No"],
                    outcome_prices=[0.3, 0.7],
                    raw_json={"id": "m2"},
                )
            ]
        return []


async def test_search_many_dedupes_without_cross_alias_hard_and() -> None:
    markets = await FakeClient().search_many(["Anthropic IPO", "Anthropic market cap"])

    assert [market.market_id for market in markets] == ["m1", "m2"]
