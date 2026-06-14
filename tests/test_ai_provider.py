from app.ai.fallback import RuleBasedProvider
from app.models import ObjectType, PointEstimate


async def test_fallback_narrative_boolean_time_numeric() -> None:
    provider = RuleBasedProvider()
    estimates = [
        PointEstimate(
            object_name="IPO",
            object_type=ObjectType.BOOLEAN,
            role="boolean",
            boolean_probability=0.75,
            anomalies=[],
            fit_method="median",
        ),
        PointEstimate(
            object_name="Timing",
            object_type=ObjectType.TIME,
            role="time",
            median_date="2026-10-31",
            anomalies=[],
            fit_method="median",
        ),
        PointEstimate(
            object_name="估值",
            object_type=ObjectType.CATEGORICAL,
            role="numeric",
            expected_value=1.6,
            unit="T",
            p50=1.6,
            anomalies=[],
            fit_method="lognormal",
        ),
    ]

    narrative = await provider.synthesize_overview("Anthropic IPO", estimates)

    assert "75%" in narrative
    assert "2026-10-31" in narrative
    assert "1.6" in narrative


async def test_fallback_narrative_categorical_only() -> None:
    provider = RuleBasedProvider()
    estimates = [
        PointEstimate(
            object_name="Top lab",
            object_type=ObjectType.CATEGORICAL,
            role="context",
            top_category="Anthropic",
            top_category_probability=0.62,
            anomalies=[],
            fit_method="argmax",
        ),
    ]

    narrative = await provider.synthesize_overview("Top lab", estimates)

    assert "Anthropic" in narrative
    assert "62%" in narrative
