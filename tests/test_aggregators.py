from app.analysis.aggregators import aggregate
from app.models import AggregationInput, ObjectType, Operator, StructuredMarket


def threshold(
    market_id: str,
    value: float,
    probability: float,
    operator: Operator = Operator.GREATER_THAN,
) -> StructuredMarket:
    return StructuredMarket(
        market_id=market_id,
        question=f"Will X be greater than {value}?",
        event_title="X",
        object_name="X",
        object_type=ObjectType.CONTINUOUS,
        operator=operator,
        threshold_value=value,
        threshold_unit="USD billion",
        category_value=None,
        probability=probability,
        probability_source="midpoint",
        resolution_date=None,
        include=True,
        confidence=0.9,
        explanation="Threshold.",
    )


def category(market_id: str, category_value: str, probability: float) -> StructuredMarket:
    return StructuredMarket(
        market_id=market_id,
        question=f"Will X be {category_value}?",
        event_title="X",
        object_name="X",
        object_type=ObjectType.CATEGORICAL,
        operator=Operator.CATEGORY,
        threshold_value=None,
        threshold_unit=None,
        category_value=category_value,
        probability=probability,
        probability_source="outcome_price",
        resolution_date=None,
        include=True,
        confidence=0.9,
        explanation="Category.",
    )


def test_continuous_thresholds_produce_interval_probabilities() -> None:
    payload = AggregationInput(
        object_name="X",
        object_type=ObjectType.CONTINUOUS,
        markets=[
            threshold("m85", 85.0, 0.70),
            threshold("m100", 100.0, 0.45),
            threshold("m150", 150.0, 0.10),
        ],
    )

    result = aggregate(payload)

    probabilities = [round(row["probability"], 2) for row in result.rows]
    assert probabilities == [0.30, 0.25, 0.35, 0.10]
    assert result.anomalies == []


def test_continuous_thresholds_flag_monotonicity_conflict() -> None:
    payload = AggregationInput(
        object_name="X",
        object_type=ObjectType.CONTINUOUS,
        markets=[
            threshold("m85", 85.0, 0.40),
            threshold("m100", 100.0, 0.45),
        ],
    )

    result = aggregate(payload)

    assert "monotonicity_conflict" in result.anomalies


def test_continuous_lower_tail_thresholds_are_aggregated() -> None:
    payload = AggregationInput(
        object_name="X",
        object_type=ObjectType.CONTINUOUS,
        markets=[
            threshold("m85", 85.0, 0.20, Operator.LESS_THAN),
            threshold("m100", 100.0, 0.50, Operator.LESS_THAN),
        ],
    )

    result = aggregate(payload)

    probabilities = [round(row["probability"], 2) for row in result.rows]
    assert probabilities == [0.20, 0.30, 0.50]
    assert result.anomalies == []


def test_categorical_money_buckets_sort_and_normalize() -> None:
    payload = AggregationInput(
        object_name="X",
        object_type=ObjectType.CATEGORICAL,
        markets=[
            category("m3", "$3.0T+", 0.065),
            category("m1", "$1.25–$1.5T", 0.11),
            category("m0", "<$1.25T", 0.07),
            category("m2", "$2.0–$2.25T", 0.385),
            category("m4", "No IPO by December 31, 2027", 0.065),
        ],
    )

    result = aggregate(payload)

    assert [row["category"] for row in result.rows] == [
        "<$1.25T",
        "$1.25–$1.5T",
        "$2.0–$2.25T",
        "$3.0T+",
        "No IPO by December 31, 2027",
    ]
    assert result.chart_json["raw_total"] == 0.695
    assert result.chart_json["normalized_total"] == 1.0
    assert "raw_probability_total_not_one" in result.anomalies
