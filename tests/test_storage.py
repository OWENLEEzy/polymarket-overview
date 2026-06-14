import sqlite3

from app.models import ObjectType, Operator, StructuredMarket
from app.storage.db import migrate
from app.storage.repositories import SearchRunRepository, StructuredMarketRepository


def test_migrate_creates_audit_tables() -> None:
    db = sqlite3.connect(":memory:")
    migrate(db)

    rows = db.execute(
        "select name from sqlite_master where type = 'table' order by name"
    ).fetchall()
    names = [row[0] for row in rows]

    assert "search_runs" in names
    assert "raw_markets" in names
    assert "structured_markets" in names
    assert "structured_market_reviews" in names
    assert "aggregation_runs" in names


def test_search_run_repository_creates_run() -> None:
    db = sqlite3.connect(":memory:")
    migrate(db)

    repo = SearchRunRepository(db)
    run_id = repo.create(topic="Anthropic IPO", provider="deepseek", search_terms=["Anthropic IPO"])

    row = db.execute("select topic, provider from search_runs where id = ?", (run_id,)).fetchone()
    assert row == ("Anthropic IPO", "deepseek")


def test_reviewed_markets_override_original_structured_markets() -> None:
    db = sqlite3.connect(":memory:")
    migrate(db)
    run_id = SearchRunRepository(db).create("X", "fallback", ["X"])
    original = make_structured_market("Original object")
    reviewed = make_structured_market("Reviewed object")

    repo = StructuredMarketRepository(db)
    repo.save_many(run_id, [original])
    repo.save_review(run_id, [reviewed])

    assert repo.list_for_run(run_id)[0].object_name == "Original object"
    assert repo.list_effective_for_run(run_id)[0].object_name == "Reviewed object"


def make_structured_market(object_name: str) -> StructuredMarket:
    return StructuredMarket(
        object_id="object-1",
        market_id="m1",
        question="Will X happen?",
        event_title="X",
        object_name=object_name,
        object_type=ObjectType.BOOLEAN,
        operator=Operator.EQUAL,
        threshold_value=None,
        threshold_unit=None,
        category_value="yes",
        probability=0.42,
        probability_source="outcome_price",
        resolution_date=None,
        include=True,
        confidence=0.8,
        explanation="Test market.",
    )
