from collections import defaultdict
import math
import re

from app.analysis.parsers import to_billions
from app.models import AggregationInput, AggregationResult, ObjectType, Operator


def aggregate(payload: AggregationInput) -> AggregationResult:
    if payload.object_type == ObjectType.TIME and all(
        market.threshold_value is None for market in payload.included_markets()
    ):
        return aggregate_categorical(payload)
    if payload.object_type in {ObjectType.CONTINUOUS, ObjectType.TIME}:
        return aggregate_continuous(payload)
    if payload.object_type in {ObjectType.CATEGORICAL, ObjectType.BOOLEAN}:
        return aggregate_categorical(payload)
    raise ValueError(f"Unsupported object type: {payload.object_type}")


def aggregate_continuous(payload: AggregationInput) -> AggregationResult:
    markets = [
        market
        for market in payload.included_markets()
        if market.threshold_value is not None
        and market.operator
        in {
            Operator.GREATER_THAN,
            Operator.GREATER_THAN_OR_EQUAL,
            Operator.LESS_THAN,
            Operator.LESS_THAN_OR_EQUAL,
        }
    ]
    points = sorted(
        [
            (
                market.threshold_value or 0.0,
                threshold_cdf_probability(market.operator, market.probability),
                market.threshold_unit,
            )
            for market in markets
        ],
        key=lambda item: item[0],
    )
    anomalies: list[str] = []
    if has_cdf_monotonicity_conflict(points):
        anomalies.append("monotonicity_conflict")

    rows: list[dict[str, float | str | None]] = []
    previous_threshold: float | None = None
    previous_cdf = 0.0

    for threshold, cdf_probability, unit_value in points:
        interval_probability = max(cdf_probability - previous_cdf, 0.0)
        unit = unit_value or ""
        label = (
            f"X <= {threshold:g} {unit}".strip()
            if previous_threshold is None
            else f"{previous_threshold:g} < X <= {threshold:g} {unit}".strip()
        )
        rows.append(
            {
                "label": label,
                "lower": previous_threshold,
                "upper": threshold,
                "probability": round(interval_probability, 6),
            }
        )
        previous_threshold = threshold
        previous_cdf = cdf_probability

    if points:
        last_threshold, last_cdf, last_unit = points[-1]
        unit = last_unit or ""
        rows.append(
            {
                "label": f"X > {last_threshold:g} {unit}".strip(),
                "lower": last_threshold,
                "upper": None,
                "probability": round(max(1.0 - last_cdf, 0.0), 6),
            }
        )

    return AggregationResult(
        object_name=payload.object_name,
        object_type=payload.object_type,
        rows=rows,
        chart_json={
            "type": "bar",
            "labels": [row["label"] for row in rows],
            "values": [row["probability"] for row in rows],
        },
        anomalies=anomalies,
    )


def threshold_cdf_probability(operator: Operator, probability: float) -> float:
    if operator in {Operator.GREATER_THAN, Operator.GREATER_THAN_OR_EQUAL}:
        return 1.0 - probability
    return probability


def has_cdf_monotonicity_conflict(points: list[tuple[float, float, str | None]]) -> bool:
    return any(
        next_probability < probability
        for (_, probability, _), (_, next_probability, _) in zip(points, points[1:], strict=False)
    )


def aggregate_categorical(payload: AggregationInput) -> AggregationResult:
    buckets: dict[str, list[float]] = defaultdict(list)
    for market in payload.included_markets():
        key = market.category_value or market.question
        buckets[key].append(market.probability)

    raw_rows = [
        (category, round(sum(values) / len(values), 6), len(values))
        for category, values in buckets.items()
    ]
    raw_total = sum(probability for _, probability, _ in raw_rows)
    rows: list[dict[str, str | float | int]] = [
        {
            "category": category,
            "probability": probability,
            "normalized_probability": round(probability / raw_total, 6) if raw_total else 0.0,
            "market_count": market_count,
        }
        for category, probability, market_count in sorted(
            raw_rows, key=lambda item: category_sort_key(item[0])
        )
    ]
    normalized_total = sum(float(row["normalized_probability"]) for row in rows)
    if rows and normalized_total != 1.0:
        rows[-1]["normalized_probability"] = round(
            float(rows[-1]["normalized_probability"]) + (1.0 - normalized_total), 6
        )

    anomalies = []
    if raw_total < 0.98 or raw_total > 1.02:
        anomalies.append("raw_probability_total_not_one")
    return AggregationResult(
        object_name=payload.object_name,
        object_type=payload.object_type,
        rows=rows,
        chart_json={
            "type": "bar",
            "labels": [row["category"] for row in rows],
            "values": [row["normalized_probability"] for row in rows],
            "raw_total": round(raw_total, 6),
            "normalized_total": round(sum(float(row["normalized_probability"]) for row in rows), 6),
        },
        anomalies=anomalies,
    )


def category_sort_key(category: str) -> tuple[int, float, str]:
    lowered = category.casefold().strip()
    if "no ipo" in lowered:
        return (1, math.inf, lowered)
    values = [
        normalize_money_value(match.group("number"), match.group("unit"))
        for match in re.finditer(
            r"\$?\s*(?P<number>\d+(?:\.\d+)?)\s*(?P<unit>t|trillion|b|bn|billion)?",
            lowered,
        )
    ]
    if lowered.startswith("<") and values:
        return (0, -math.inf, lowered)
    if values:
        return (0, min(values), lowered)
    return (2, math.inf, lowered)


def normalize_money_value(number_text: str, unit_text: str | None) -> float:
    return to_billions(float(number_text), unit_text or "B")
