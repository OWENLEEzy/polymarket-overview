from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app
from app.models import ObjectType, Operator, StructuredMarket
from app.storage.db import connect, migrate
from app.storage.repositories import SearchRunRepository, StructuredMarketRepository


def test_search_page_loads() -> None:
    client = TestClient(app)

    response = client.get("/")

    assert response.status_code == 200
    assert "Polymarket Overview" in response.text
    assert 'name="topic"' in response.text


def test_review_route_saves_corrected_structure(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "test.sqlite"
    monkeypatch.setenv("POLYMARKET_OVERVIEW_DB_PATH", str(db_path))
    get_settings.cache_clear()
    db = connect(str(db_path))
    migrate(db)
    run_id = SearchRunRepository(db).create("X", "fallback", ["X"])
    StructuredMarketRepository(db).save_many(run_id, [make_market()])
    client = TestClient(app)

    response = client.post(
        f"/review/{run_id}",
        data={
            "include_0": "on",
            "event_title_0": "X",
            "object_name_0": "Corrected X",
            "object_type_0": "boolean",
            "operator_0": "=",
            "threshold_value_0": "",
            "threshold_unit_0": "",
            "category_value_0": "yes",
            "resolution_date_0": "",
        },
        follow_redirects=False,
    )

    assert response.status_code == 303
    reviewed = StructuredMarketRepository(db).list_effective_for_run(run_id)
    assert reviewed[0].object_name == "Corrected X"
    assert reviewed[0].probability == 0.42
    get_settings.cache_clear()


def make_market() -> StructuredMarket:
    return StructuredMarket(
        object_id="x",
        market_id="m1",
        question="Will X happen?",
        event_title="X",
        object_name="Original X",
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


def _seed_run(tmp_path, monkeypatch):
    monkeypatch.setenv("POLYMARKET_OVERVIEW_DB_PATH", str(tmp_path / "w.sqlite"))
    get_settings.cache_clear()
    db = connect(get_settings().db_path)
    migrate(db)
    run_id = SearchRunRepository(db).create("Anthropic valuation", "fallback", ["x"])
    StructuredMarketRepository(db).save_many(
        run_id,
        [
            StructuredMarket(
                object_id="v",
                market_id="a",
                question="Q?",
                event_title="E",
                object_name="Valuation",
                object_type=ObjectType.CATEGORICAL,
                operator=Operator.CATEGORY,
                threshold_value=None,
                threshold_unit=None,
                category_value="$1.25–$1.5T",
                probability=0.5,
                probability_source="outcome_price",
                resolution_date=None,
                include=True,
                confidence=0.9,
                explanation="t",
            ),
            StructuredMarket(
                object_id="v",
                market_id="b",
                question="Q?",
                event_title="E",
                object_name="Valuation",
                object_type=ObjectType.CATEGORICAL,
                operator=Operator.CATEGORY,
                threshold_value=None,
                threshold_unit=None,
                category_value="$1.5–$2.25T",
                probability=0.5,
                probability_source="outcome_price",
                resolution_date=None,
                include=True,
                confidence=0.9,
                explanation="t",
            ),
        ],
    )
    return run_id


def test_summarize_endpoint_returns_json(tmp_path, monkeypatch) -> None:
    run_id = _seed_run(tmp_path, monkeypatch)
    client = TestClient(app)

    response = client.post(f"/summarize/{run_id}")

    assert response.status_code == 200
    body = response.json()
    assert body["topic"] == "Anthropic valuation"
    assert body["narrative"]
    assert body["point_estimates"][0]["role"] == "numeric"
    get_settings.cache_clear()


def test_aggregate_endpoint_returns_json(tmp_path, monkeypatch) -> None:
    run_id = _seed_run(tmp_path, monkeypatch)
    client = TestClient(app)

    response = client.get(f"/aggregate/{run_id}/v")

    assert response.status_code == 200
    assert response.json()["object_name"] == "Valuation"
    get_settings.cache_clear()


def test_summarize_unknown_run_returns_404(tmp_path, monkeypatch) -> None:
    _seed_run(tmp_path, monkeypatch)
    client = TestClient(app)

    response = client.post("/summarize/does-not-exist")

    assert response.status_code == 404
    get_settings.cache_clear()


def test_aggregate_unknown_object_returns_404(tmp_path, monkeypatch) -> None:
    run_id = _seed_run(tmp_path, monkeypatch)
    client = TestClient(app)

    response = client.get(f"/aggregate/{run_id}/nonexistent")

    assert response.status_code == 404
    get_settings.cache_clear()
