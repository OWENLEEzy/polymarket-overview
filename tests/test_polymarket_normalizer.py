from app.polymarket.normalizer import normalize_market


def test_normalize_gamma_market_parses_outcomes_prices_and_tokens() -> None:
    payload = {
        "id": "123",
        "conditionId": "cond-1",
        "question": "Will Anthropic IPO above $100B?",
        "outcomes": '["Yes","No"]',
        "outcomePrices": '["0.42","0.58"]',
        "clobTokenIds": '["token-yes","token-no"]',
        "closed": False,
        "archived": False,
        "liquidity": "1020.5",
        "volume": "991.2",
        "events": [{"id": "e1", "title": "Anthropic IPO"}],
    }

    market = normalize_market(payload)

    assert market.market_id == "123"
    assert market.event_id == "e1"
    assert market.outcomes == ["Yes", "No"]
    assert market.outcome_prices == [0.42, 0.58]
    assert market.token_ids == ["token-yes", "token-no"]
