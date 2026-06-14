from app.analysis.objects import build_objects
from app.analysis.probability import probability_from_outcomes
from app.models import ObjectType, Operator, RawMarket, StructuredMarket


def _raw(**kw: object) -> RawMarket:
    base: dict[str, object] = {
        "market_id": "m1",
        "question": "Will X be above $100B?",
        "event_title": "X IPO",
        "outcomes": ["Yes", "No"],
        "outcome_prices": [0.37, 0.63],
        "raw_json": {"id": "m1"},
    }
    base.update(kw)
    return RawMarket.model_validate(base)


def test_probability_uses_yes_leg() -> None:
    assert probability_from_outcomes(_raw(), None) == 0.37


def test_probability_matches_category_outcome() -> None:
    raw = _raw(
        question="Which valuation bucket?",
        outcomes=["< $50B", "$50B-$100B", "> $100B"],
        outcome_prices=[0.2, 0.5, 0.3],
    )
    assert probability_from_outcomes(raw, "$50B-$100B") == 0.5


def test_probability_none_when_no_prices() -> None:
    assert probability_from_outcomes(_raw(outcome_prices=[]), None) is None


def test_probability_falls_back_to_first_price() -> None:
    raw = _raw(outcomes=["Up", "Down"], outcome_prices=[0.6, 0.4])
    assert probability_from_outcomes(raw, None) == 0.6


def _structured(object_id: str, market_id: str) -> StructuredMarket:
    return StructuredMarket(
        object_id=object_id,
        market_id=market_id,
        question="q",
        event_title="E",
        object_name="Obj",
        object_type=ObjectType.CONTINUOUS,
        operator=Operator.GREATER_THAN_OR_EQUAL,
        threshold_value=100.0,
        threshold_unit="B",
        category_value=None,
        probability=0.5,
        probability_source="outcome_price",
        resolution_date=None,
        include=True,
        confidence=1.0,
        explanation="x",
    )


def test_build_objects_rolls_up_one_per_object_id() -> None:
    markets = [_structured("obj-1", "m1"), _structured("obj-1", "m2")]
    objects = build_objects(markets)
    assert len(objects) == 1
    assert objects[0].object_id == "obj-1"
    assert objects[0].market_ids == ["m1", "m2"]
