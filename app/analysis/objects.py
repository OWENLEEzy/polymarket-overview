from collections import defaultdict

from app.models import DataObjectCandidate, StructuredMarket


def build_objects(markets: list[StructuredMarket]) -> list[DataObjectCandidate]:
    """Roll finalized structured markets up into one object candidate per object_id."""
    grouped: dict[str, list[StructuredMarket]] = defaultdict(list)
    for market in markets:
        grouped[market.object_id or market.object_name].append(market)

    objects: list[DataObjectCandidate] = []
    for object_id, members in grouped.items():
        first = members[0]
        market_ids = list(dict.fromkeys(market.market_id for market in members))
        confidence = sum(market.confidence for market in members) / len(members)
        objects.append(
            DataObjectCandidate(
                object_id=object_id,
                object_name=first.object_name,
                object_type=first.object_type,
                market_ids=market_ids,
                confidence=round(confidence, 6),
                explanation="Deterministic grouping from finalized structured markets.",
            )
        )
    return sorted(objects, key=lambda item: item.object_name)
