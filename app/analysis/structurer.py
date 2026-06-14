import re
from collections import defaultdict
from dataclasses import dataclass

from app.analysis.objects import build_objects
from app.analysis.parsers import (
    MoneyBracket,
    canonical_unit,
    parse_deadline_date,
    parse_money_bracket,
)
from app.analysis.probability import probability_from_outcomes
from app.models import (
    ObjectType,
    Operator,
    RawMarket,
    StructuredExtraction,
    StructuredMarket,
    build_object_id,
)

_YEAR_RE = re.compile(r"\b\d{4}\b")


def _group_key(market: RawMarket) -> str:
    if market.event_id:
        return f"event:{market.event_id}"
    if market.event_title:
        return f"title:{market.event_title.strip().casefold()}"
    return f"market:{market.market_id}"


def group_markets(markets: list[RawMarket]) -> dict[str, list[RawMarket]]:
    """Group raw markets by their Polymarket event identity (the source of truth
    for which markets belong to one data object), falling back to event title,
    then to the market id when nothing else is available."""
    groups: dict[str, list[RawMarket]] = defaultdict(list)
    for market in markets:
        groups[_group_key(market)].append(market)
    return dict(groups)


@dataclass(frozen=True)
class ClassifiedMarket:
    raw: RawMarket
    operator: Operator
    threshold_value: float | None
    threshold_unit: str | None
    category_value: str | None
    resolution_date: str | None


def _bracket_of(market: RawMarket) -> MoneyBracket | None:
    return parse_money_bracket(market.group_item_title) or parse_money_bracket(market.question)


def _date_of(market: RawMarket) -> str | None:
    if market.end_date:
        return market.end_date.split("T")[0]
    return parse_deadline_date(market.question)


def _looks_like_date_bucket(market: RawMarket) -> bool:
    # Require a 4-digit year so fuzzy date parsing does not misread a candidate
    # name (e.g. "Marco Rubio") as a date and wrongly route a categorical group
    # to TIME.
    title = market.group_item_title or ""
    return bool(_YEAR_RE.search(title)) and parse_deadline_date(title) is not None


def _bracket_operator(bracket: MoneyBracket) -> tuple[Operator, float]:
    # Open-upper "$100B+" -> threshold is the lower edge, operator >=.
    if bracket.lower is not None and bracket.upper is None:
        return Operator.GREATER_THAN_OR_EQUAL, bracket.lower
    # Open-lower "<$100B" -> threshold is the upper edge, operator <.
    if bracket.lower is None and bracket.upper is not None:
        return Operator.LESS_THAN, bracket.upper
    # Closed range "$100B-$150B" -> keep the lower edge as the representative point.
    if bracket.lower is not None and bracket.upper is not None and bracket.lower != bracket.upper:
        return Operator.RANGE, bracket.lower
    # Single "$100B".
    value = bracket.lower if bracket.lower is not None else bracket.upper
    return Operator.GREATER_THAN_OR_EQUAL, float(value or 0.0)


def classify_group(
    topic: str, markets: list[RawMarket]
) -> tuple[ObjectType, list[ClassifiedMarket]]:
    """Infer one object_type for a group, then derive each member's axis fields."""
    brackets = {m.market_id: _bracket_of(m) for m in markets}
    numeric = [m for m in markets if brackets[m.market_id] is not None]
    date_bucketed = [m for m in markets if _looks_like_date_bucket(m)]
    named = [
        m
        for m in markets
        if m.group_item_title and brackets[m.market_id] is None and not _looks_like_date_bucket(m)
    ]

    if len(numeric) >= 2 or (numeric and len(markets) == len(numeric)):
        object_type = ObjectType.CONTINUOUS
    elif len(date_bucketed) >= 2:
        object_type = ObjectType.TIME
    elif len(named) >= 2 or any(len(m.outcomes) > 2 for m in markets):
        object_type = ObjectType.CATEGORICAL
    else:
        object_type = ObjectType.BOOLEAN

    classified = [
        _classify_member(object_type, market, brackets[market.market_id]) for market in markets
    ]
    return object_type, classified


def _classify_member(
    object_type: ObjectType, market: RawMarket, bracket: MoneyBracket | None
) -> ClassifiedMarket:
    if object_type == ObjectType.CONTINUOUS and bracket is not None:
        operator, threshold = _bracket_operator(bracket)
        return ClassifiedMarket(
            raw=market,
            operator=operator,
            threshold_value=threshold,
            threshold_unit=canonical_unit(bracket.unit),
            category_value=None,
            resolution_date=_date_of(market),
        )
    if object_type == ObjectType.TIME:
        return ClassifiedMarket(
            raw=market,
            operator=Operator.CATEGORY,
            threshold_value=None,
            threshold_unit=None,
            category_value=None,
            resolution_date=_date_of(market),
        )
    if object_type == ObjectType.CATEGORICAL:
        return ClassifiedMarket(
            raw=market,
            operator=Operator.CATEGORY,
            threshold_value=None,
            threshold_unit=None,
            category_value=market.group_item_title or market.question,
            resolution_date=_date_of(market),
        )
    return ClassifiedMarket(
        raw=market,
        operator=Operator.EQUAL,
        threshold_value=None,
        threshold_unit=None,
        category_value="yes",
        resolution_date=_date_of(market),
    )


def _object_name(markets: list[RawMarket], topic: str) -> str:
    title = next((m.event_title for m in markets if m.event_title), None)
    return title or topic


def structure_markets(topic: str, raw_markets: list[RawMarket]) -> StructuredExtraction:
    """The single deterministic source of truth for structuring.

    Groups markets by event identity, classifies each group's object_type, derives
    operator/threshold/date/category from Polymarket's own fields via the existing
    parsers, and joins probability straight from outcome prices. No LLM, no network,
    so the token-truncation failure mode is impossible by construction.
    """
    groups = group_markets(raw_markets)
    structured: list[StructuredMarket] = []

    for members in groups.values():
        object_type, classified = classify_group(topic, members)
        object_name = _object_name(members, topic)
        # One object_id per group: pass None for the per-market-varying fields so a
        # ladder of differing thresholds/dates co-groups instead of fragmenting.
        object_id = build_object_id(object_name, object_type, members[0].event_title, None, None)

        for item in classified:
            probability = probability_from_outcomes(item.raw, item.category_value)
            if probability is None:
                continue
            structured.append(
                StructuredMarket(
                    object_id=object_id,
                    market_id=item.raw.market_id,
                    question=item.raw.question,
                    event_title=item.raw.event_title or object_name,
                    object_name=object_name,
                    object_type=object_type,
                    operator=item.operator,
                    threshold_value=item.threshold_value,
                    threshold_unit=item.threshold_unit,
                    category_value=item.category_value,
                    probability=probability,
                    probability_source="outcome_price",
                    resolution_date=item.resolution_date,
                    include=not item.raw.closed and not item.raw.archived,
                    confidence=1.0,
                    explanation=f"Deterministic: {object_type.value} via {item.operator.value}",
                )
            )

    return StructuredExtraction(
        topic=topic,
        objects=build_objects(structured),
        markets=structured,
    )
