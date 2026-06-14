import pytest

from app.analysis.point_estimate import (
    Interval,
    argmax_estimate,
    boolean_estimate,
    compute_point_estimate,
    median_date_estimate,
    numeric_estimate,
)
from app.models import ObjectType, Operator, StructuredMarket


def _market(**overrides) -> StructuredMarket:
    base = dict(
        object_id="obj",
        market_id="m",
        question="Q?",
        event_title="E",
        object_name="Obj",
        object_type=ObjectType.CATEGORICAL,
        operator=Operator.CATEGORY,
        threshold_value=None,
        threshold_unit=None,
        category_value=None,
        probability=0.5,
        probability_source="outcome_price",
        resolution_date=None,
        include=True,
        confidence=0.9,
        explanation="t",
    )
    base.update(overrides)
    return StructuredMarket(**base)


def test_lognormal_fit_recovers_central_value() -> None:
    # Three CDF points consistent with a lognormal centered near 1500B.
    intervals = [
        Interval(lower=None, upper=1250.0, probability=0.15),
        Interval(lower=1250.0, upper=1500.0, probability=0.35),
        Interval(lower=1500.0, upper=2250.0, probability=0.35),
        Interval(lower=2250.0, upper=None, probability=0.15),
    ]
    result = numeric_estimate(intervals, unit="B")
    assert result.fit_method == "lognormal"
    assert 1200.0 < result.expected_value < 2000.0
    assert result.p10 < result.p50 < result.p90


def test_falls_back_to_midpoint_with_too_few_points() -> None:
    intervals = [
        Interval(lower=None, upper=1000.0, probability=0.4),
        Interval(lower=1000.0, upper=None, probability=0.6),
    ]
    result = numeric_estimate(intervals, unit="B")
    assert result.fit_method == "midpoint"
    assert "lognormal_fit_failed" in result.anomalies
    # The 0.6 open upper tail also legitimately trips the high-tail check; assert
    # it so the anomaly set is fully pinned and the intent is unambiguous.
    assert "high_tail_probability" in result.anomalies
    assert result.expected_value > 0


def test_flags_high_tail_probability() -> None:
    intervals = [
        Interval(lower=None, upper=1000.0, probability=0.3),
        Interval(lower=1000.0, upper=2000.0, probability=0.3),
        Interval(lower=2000.0, upper=None, probability=0.4),
    ]
    result = numeric_estimate(intervals, unit="B")
    assert "high_tail_probability" in result.anomalies


def test_flags_insufficient_data() -> None:
    intervals = [Interval(lower=None, upper=1000.0, probability=1.0)]
    result = numeric_estimate(intervals, unit="B")
    assert "insufficient_data" in result.anomalies
    assert result.expected_value is None


def test_boolean_returns_median_probability() -> None:
    value, anomalies = boolean_estimate([0.70, 0.75, 0.80])
    assert value == 0.75
    assert anomalies == []


def test_boolean_flags_heterogeneous_group() -> None:
    _, anomalies = boolean_estimate([0.10, 0.50, 0.95])
    assert "heterogeneous_group" in anomalies


def test_argmax_returns_top_category() -> None:
    top, probability = argmax_estimate({"Anthropic": [0.30, 0.40], "OpenAI": [0.10]})
    assert top == "Anthropic"
    assert probability == pytest.approx(0.7777, abs=1e-3)


def test_median_date_returns_first_over_half() -> None:
    date, anomalies = median_date_estimate(
        [("2026-06-30", 0.2), ("2026-10-31", 0.4), ("2027-03-31", 0.3)]
    )
    assert date == "2026-10-31"
    assert anomalies == []


def test_median_date_not_reached() -> None:
    date, anomalies = median_date_estimate([("2026-06-30", 0.2), ("2026-10-31", 0.2)])
    assert date == "2026-10-31"
    assert "median_date_not_reached" in anomalies


def test_categorical_money_buckets_route_to_numeric() -> None:
    markets = [
        _market(market_id="a", category_value="<$1.25T", probability=0.15),
        _market(market_id="b", category_value="$1.25–$1.5T", probability=0.35),
        _market(market_id="c", category_value="$1.5–$2.25T", probability=0.35),
        _market(market_id="d", category_value="$3.0T+", probability=0.10),
        _market(market_id="e", category_value="No IPO", probability=0.05),
    ]
    estimate = compute_point_estimate("Valuation", ObjectType.CATEGORICAL, markets)
    assert estimate.role == "numeric"
    assert estimate.expected_value is not None
    assert estimate.unit == "T"


def test_text_categorical_routes_to_argmax() -> None:
    markets = [
        _market(market_id="a", category_value="Anthropic", probability=0.6),
        _market(market_id="b", category_value="OpenAI", probability=0.3),
    ]
    estimate = compute_point_estimate("Top lab", ObjectType.CATEGORICAL, markets)
    assert estimate.role == "context"
    assert estimate.top_category == "Anthropic"
    assert estimate.fit_method == "argmax"


def test_boolean_routes_to_median() -> None:
    markets = [
        _market(
            object_type=ObjectType.BOOLEAN,
            operator=Operator.EQUAL,
            category_value="yes",
            probability=0.75,
        )
    ]
    estimate = compute_point_estimate("IPO", ObjectType.BOOLEAN, markets)
    assert estimate.role == "boolean"
    assert estimate.boolean_probability == 0.75


def test_time_uses_resolution_date() -> None:
    markets = [
        _market(
            market_id="a",
            object_type=ObjectType.TIME,
            category_value="by Q2",
            probability=0.3,
            resolution_date="2026-06-30",
        ),
        _market(
            market_id="b",
            object_type=ObjectType.TIME,
            category_value="by Q4",
            probability=0.5,
            resolution_date="2026-12-31",
        ),
    ]
    estimate = compute_point_estimate("Timing", ObjectType.TIME, markets)
    assert estimate.role == "time"
    assert estimate.median_date == "2026-12-31"


def test_continuous_thresholds_flag_monotonicity_conflict() -> None:
    # Higher threshold with higher P(>x) means the implied CDF decreases — a
    # real Polymarket inconsistency the detail view already flags. The point
    # estimate must surface it too, so the narrative can hedge.
    markets = [
        _market(
            market_id="a",
            object_type=ObjectType.CONTINUOUS,
            operator=Operator.GREATER_THAN,
            threshold_value=85.0,
            threshold_unit="USD billion",
            probability=0.40,
        ),
        _market(
            market_id="b",
            object_type=ObjectType.CONTINUOUS,
            operator=Operator.GREATER_THAN,
            threshold_value=100.0,
            threshold_unit="USD billion",
            probability=0.45,
        ),
    ]
    estimate = compute_point_estimate("Valuation", ObjectType.CONTINUOUS, markets)
    assert "monotonicity_conflict" in estimate.anomalies


def test_continuous_thresholds_route_to_numeric() -> None:
    markets = [
        _market(
            market_id="a",
            object_type=ObjectType.CONTINUOUS,
            operator=Operator.GREATER_THAN,
            threshold_value=85.0,
            threshold_unit="USD billion",
            probability=0.70,
        ),
        _market(
            market_id="b",
            object_type=ObjectType.CONTINUOUS,
            operator=Operator.GREATER_THAN,
            threshold_value=100.0,
            threshold_unit="USD billion",
            probability=0.45,
        ),
        _market(
            market_id="c",
            object_type=ObjectType.CONTINUOUS,
            operator=Operator.GREATER_THAN,
            threshold_value=150.0,
            threshold_unit="USD billion",
            probability=0.10,
        ),
    ]
    estimate = compute_point_estimate("Valuation", ObjectType.CONTINUOUS, markets)
    assert estimate.role == "numeric"
    assert estimate.unit == "B"
