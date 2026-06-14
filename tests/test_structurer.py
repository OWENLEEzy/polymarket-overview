from app.analysis.structurer import classify_group, group_markets, structure_markets
from app.models import ObjectType, Operator, RawMarket


def _raw(market_id: str, event_id: str | None, **kw: object) -> RawMarket:
    base: dict[str, object] = {
        "market_id": market_id,
        "event_id": event_id,
        "question": kw.get("question", f"q {market_id}"),
        "event_title": kw.get("event_title", "Event"),
        "outcomes": kw.get("outcomes", ["Yes", "No"]),
        "outcome_prices": kw.get("outcome_prices", [0.5, 0.5]),
        "raw_json": {"id": market_id},
    }
    base.update({k: v for k, v in kw.items() if k in RawMarket.model_fields})
    return RawMarket.model_validate(base)


# --- grouping -------------------------------------------------------------


def test_group_markets_groups_by_event_id() -> None:
    markets = [_raw("m1", "e1"), _raw("m2", "e1"), _raw("m3", "e2")]
    groups = group_markets(markets)
    assert sorted(len(members) for members in groups.values()) == [1, 2]


def test_group_markets_falls_back_to_event_title_then_market_id() -> None:
    markets = [
        _raw("m1", None, event_title="Shared"),
        _raw("m2", None, event_title="Shared"),
        _raw("m3", None, event_title=None),
    ]
    groups = group_markets(markets)
    assert sorted(len(members) for members in groups.values()) == [1, 2]


# --- classification -------------------------------------------------------


def test_classify_continuous_ladder() -> None:
    markets = [
        _raw("m1", "e1", group_item_title=">$100B", question="Above $100B?"),
        _raw("m2", "e1", group_item_title=">$150B", question="Above $150B?"),
    ]
    object_type, members = classify_group("Anthropic", markets)
    assert object_type == ObjectType.CONTINUOUS
    assert {m.operator for m in members} == {Operator.GREATER_THAN_OR_EQUAL}
    assert sorted(m.threshold_value for m in members if m.threshold_value is not None) == [
        100.0,
        150.0,
    ]


def test_classify_categorical_candidates() -> None:
    markets = [
        _raw("m1", "e1", group_item_title="Marco Rubio", question="Rubio nominee?"),
        _raw("m2", "e1", group_item_title="Nikki Haley", question="Haley nominee?"),
    ]
    object_type, members = classify_group("2028 GOP", markets)
    assert object_type == ObjectType.CATEGORICAL
    assert {m.category_value for m in members} == {"Marco Rubio", "Nikki Haley"}
    assert all(m.operator == Operator.CATEGORY for m in members)


def test_classify_time_ladder_prefers_end_date() -> None:
    markets = [
        _raw(
            "m1",
            "e1",
            group_item_title="June 2026",
            question="IPO by 30 June 2026?",
            end_date="2026-06-30T00:00:00Z",
        ),
        _raw(
            "m2",
            "e1",
            group_item_title="December 2026",
            question="IPO by 31 December 2026?",
            end_date="2026-12-31T00:00:00Z",
        ),
    ]
    object_type, members = classify_group("Anthropic IPO", markets)
    assert object_type == ObjectType.TIME
    assert {m.resolution_date for m in members} == {"2026-06-30", "2026-12-31"}


def test_classify_lone_boolean() -> None:
    markets = [_raw("m1", "e1", question="Will a recession happen in 2026?", group_item_title=None)]
    object_type, members = classify_group("Recession", markets)
    assert object_type == ObjectType.BOOLEAN
    assert members[0].operator == Operator.EQUAL


# --- assembly -------------------------------------------------------------


def test_structure_markets_builds_objects_and_joins_probability() -> None:
    markets = [
        _raw(
            "m1",
            "e1",
            group_item_title=">$100B",
            question="Above $100B?",
            outcomes=["Yes", "No"],
            outcome_prices=[0.4, 0.6],
        ),
        _raw(
            "m2",
            "e1",
            group_item_title=">$150B",
            question="Above $150B?",
            outcomes=["Yes", "No"],
            outcome_prices=[0.2, 0.8],
        ),
    ]
    extraction = structure_markets("Anthropic", markets)

    assert len(extraction.objects) == 1
    assert extraction.objects[0].object_type == ObjectType.CONTINUOUS
    assert sorted(m.probability for m in extraction.markets) == [0.2, 0.4]
    # All members share one object_id: a ladder must not fragment into singletons.
    assert len({m.object_id for m in extraction.markets}) == 1
    assert extraction.objects[0].object_id == extraction.markets[0].object_id


def test_structure_markets_separates_distinct_events() -> None:
    markets = [
        _raw(
            "m1",
            "e1",
            event_title="Anthropic IPO",
            question="IPO above $100B?",
            group_item_title=">$100B",
        ),
        _raw(
            "m2",
            "e2",
            event_title="OpenAI IPO",
            question="IPO above $200B?",
            group_item_title=">$200B",
        ),
    ]
    extraction = structure_markets("AI labs", markets)
    assert len({m.object_id for m in extraction.markets}) == 2


def test_structure_markets_boolean_fallback_included() -> None:
    # Free-text Yes/No with no bracket and no date => boolean, still included.
    markets = [_raw("m1", "e1", question="Will it rain?", group_item_title=None)]
    extraction = structure_markets("Weather", markets)
    assert extraction.markets[0].object_type == ObjectType.BOOLEAN
    assert extraction.markets[0].include is True


def test_structure_markets_excludes_closed_markets() -> None:
    markets = [_raw("m1", "e1", question="Resolved?", closed=True)]
    extraction = structure_markets("X", markets)
    assert extraction.markets[0].include is False


def test_structure_markets_drops_markets_without_prices() -> None:
    markets = [_raw("m1", "e1", question="No prices?", outcome_prices=[])]
    extraction = structure_markets("X", markets)
    assert extraction.markets == []
