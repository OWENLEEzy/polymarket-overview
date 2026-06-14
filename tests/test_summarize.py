from app.config import get_settings
from app.models import ObjectType, Operator, StructuredMarket
from app.service import OverviewService
from app.storage.db import connect, migrate
from app.storage.repositories import SearchRunRepository, StructuredMarketRepository


def _market(object_id: str, category_value: str, probability: float) -> StructuredMarket:
    return StructuredMarket(
        object_id=object_id,
        market_id=category_value,
        question="Q?",
        event_title="E",
        object_name="Valuation",
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
        explanation="t",
    )


async def test_summarize_returns_narrative_and_estimates(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("POLYMARKET_OVERVIEW_DB_PATH", str(tmp_path / "s.sqlite"))
    get_settings.cache_clear()
    db = connect(get_settings().db_path)
    migrate(db)
    run_id = SearchRunRepository(db).create("Anthropic valuation", "fallback", ["x"])
    StructuredMarketRepository(db).save_many(
        run_id,
        [
            _market("v", "<$1.25T", 0.15),
            _market("v", "$1.25–$1.5T", 0.35),
            _market("v", "$1.5–$2.25T", 0.35),
            _market("v", "$3.0T+", 0.15),
        ],
    )

    service = OverviewService(get_settings())
    summary = await service.summarize(run_id)

    assert summary.topic == "Anthropic valuation"
    assert summary.narrative
    assert summary.point_estimates[0].role == "numeric"
    get_settings.cache_clear()
