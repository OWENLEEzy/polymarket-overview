from app.models import RawMarket


def probability_from_outcomes(market: RawMarket, category_value: str | None) -> float | None:
    """Read the market's probability straight from Polymarket outcome prices.

    Prefer the price of the outcome whose label matches ``category_value``; fall
    back to the "Yes" leg, then to the first price. Returns ``None`` when the
    market carries no prices at all (such markets cannot contribute to any
    aggregate and are dropped upstream).
    """
    if not market.outcome_prices:
        return None
    if category_value and market.outcomes:
        wanted = category_value.casefold()
        for outcome, price in zip(market.outcomes, market.outcome_prices, strict=False):
            if outcome.casefold() == wanted:
                return price
    if market.outcomes:
        for outcome, price in zip(market.outcomes, market.outcome_prices, strict=False):
            if outcome.casefold() == "yes":
                return price
    return market.outcome_prices[0]
