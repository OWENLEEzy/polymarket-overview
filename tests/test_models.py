from app.models import (
    AggregationInput,
    ObjectType,
    Operator,
    OverviewSummary,
    PointEstimate,
    StructuredMarket,
)


def test_structured_market_accepts_continuous_threshold() -> None:
    market = StructuredMarket(
        market_id="m1",
        question="Will Anthropic IPO above $100B?",
        event_title="Anthropic IPO valuation",
        object_name="Anthropic IPO valuation",
        object_type=ObjectType.CONTINUOUS,
        operator=Operator.GREATER_THAN,
        threshold_value=100.0,
        threshold_unit="USD billion",
        category_value=None,
        probability=0.42,
        probability_source="midpoint",
        resolution_date="2027-12-31",
        include=True,
        confidence=0.91,
        explanation="Threshold market for IPO valuation.",
    )

    assert market.threshold_value == 100.0
    assert market.probability == 0.42


def test_aggregation_input_filters_included_markets() -> None:
    included = StructuredMarket(
        market_id="m1",
        question="Will X be above 10?",
        event_title="X",
        object_name="X",
        object_type=ObjectType.CONTINUOUS,
        operator=Operator.GREATER_THAN,
        threshold_value=10.0,
        threshold_unit="units",
        category_value=None,
        probability=0.7,
        probability_source="midpoint",
        resolution_date=None,
        include=True,
        confidence=0.8,
        explanation="Included.",
    )
    excluded = included.model_copy(update={"market_id": "m2", "include": False})

    payload = AggregationInput(
        object_name="X", object_type=ObjectType.CONTINUOUS, markets=[included, excluded]
    )

    assert [market.market_id for market in payload.included_markets()] == ["m1"]


def test_point_estimate_defaults_optional_fields() -> None:
    estimate = PointEstimate(
        object_name="Anthropic IPO",
        object_type=ObjectType.BOOLEAN,
        role="boolean",
        boolean_probability=0.75,
        anomalies=[],
        fit_method="median",
    )
    assert estimate.boolean_probability == 0.75
    assert estimate.expected_value is None


def test_overview_summary_holds_estimates() -> None:
    summary = OverviewSummary(
        search_run_id="run-1",
        topic="Anthropic IPO",
        narrative="市场预测发生概率为 75%。",
        point_estimates=[],
    )
    assert summary.topic == "Anthropic IPO"
    assert summary.point_estimates == []
