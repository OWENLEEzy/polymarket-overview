from app.models import Operator, StructuredMarket


def has_survival_monotonicity_conflict(markets: list[StructuredMarket]) -> bool:
    points = sorted(
        [
            (market.threshold_value, market.probability)
            for market in markets
            if market.threshold_value is not None
            and market.operator in {Operator.GREATER_THAN, Operator.GREATER_THAN_OR_EQUAL}
        ],
        key=lambda item: item[0],
    )
    return any(
        next_probability > probability
        for (_, probability), (_, next_probability) in zip(points, points[1:], strict=False)
    )
