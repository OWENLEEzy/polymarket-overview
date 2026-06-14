from app.models import RawMarket
from app.recall.ranker import rank_markets


def make_market(market_id: str, question: str) -> RawMarket:
    return RawMarket(
        market_id=market_id,
        question=question,
        event_title="Test",
        outcomes=["Yes", "No"],
        outcome_prices=[0.5, 0.5],
        raw_json={"id": market_id, "question": question},
    )


def test_rank_markets_prioritizes_query_terms() -> None:
    markets = [
        make_market("m1", "Will Anthropic IPO above $100B?"),
        make_market("m2", "Will Bitcoin hit $100k?"),
    ]

    ranked = rank_markets("Anthropic IPO valuation", markets)

    assert ranked[0].market.market_id == "m1"
    assert ranked[0].score > ranked[1].score
