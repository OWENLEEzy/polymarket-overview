import json
from typing import Any

from app.models import RawMarket


def parse_json_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        parsed = json.loads(value)
        return parsed if isinstance(parsed, list) else []
    return []


def parse_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_market(payload: dict[str, Any]) -> RawMarket:
    events = payload.get("events")
    event = events[0] if isinstance(events, list) and events else {}
    outcomes = [str(item) for item in parse_json_list(payload.get("outcomes"))]
    prices = [float(item) for item in parse_json_list(payload.get("outcomePrices"))]
    tokens = [str(item) for item in parse_json_list(payload.get("clobTokenIds"))]

    market_id = payload.get("id") or payload.get("conditionId") or payload.get("slug")
    if market_id is None:
        raise ValueError("Market payload has no id, conditionId, or slug")

    return RawMarket(
        market_id=str(market_id),
        event_id=str(event.get("id")) if event.get("id") is not None else None,
        question=str(payload.get("question") or ""),
        event_title=str(event.get("title")) if event.get("title") is not None else None,
        slug=str(payload.get("slug")) if payload.get("slug") is not None else None,
        end_date=str(payload.get("endDate")) if payload.get("endDate") is not None else None,
        group_item_title=(
            str(payload.get("groupItemTitle"))
            if payload.get("groupItemTitle") is not None
            else None
        ),
        group_item_range=[str(item) for item in parse_json_list(payload.get("groupItemRange"))],
        outcomes=outcomes,
        outcome_prices=prices,
        token_ids=tokens,
        closed=bool(payload.get("closed", False)),
        archived=bool(payload.get("archived", False)),
        active=bool(payload.get("active", True)),
        liquidity=parse_float(payload.get("liquidity")),
        volume=parse_float(payload.get("volume")),
        raw_json=payload,
    )


def dedupe_markets(markets: list[RawMarket]) -> list[RawMarket]:
    seen: set[str] = set()
    deduped: list[RawMarket] = []
    for market in markets:
        if market.market_id in seen:
            continue
        seen.add(market.market_id)
        deduped.append(market)
    return deduped
