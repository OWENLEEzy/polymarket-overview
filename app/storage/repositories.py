import json
import sqlite3
from datetime import UTC, datetime
from uuid import uuid4

from app.models import AggregationInput, AggregationResult, RawMarket, StructuredMarket


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


class SearchRunRepository:
    def __init__(self, db: sqlite3.Connection) -> None:
        self.db = db

    def create(self, topic: str, provider: str, search_terms: list[str]) -> str:
        run_id = str(uuid4())
        self.db.execute(
            "insert into search_runs (id, topic, provider, search_terms_json, created_at) values (?, ?, ?, ?, ?)",
            (run_id, topic, provider, json.dumps(search_terms), now_iso()),
        )
        self.db.commit()
        return run_id

    def get_topic(self, search_run_id: str) -> str | None:
        row = self.db.execute(
            "select topic from search_runs where id = ?",
            (search_run_id,),
        ).fetchone()
        return str(row[0]) if row else None


class RawMarketRepository:
    def __init__(self, db: sqlite3.Connection) -> None:
        self.db = db

    def save_many(self, search_run_id: str, markets: list[RawMarket]) -> None:
        self.db.executemany(
            "insert into raw_markets (id, search_run_id, market_id, payload_json, created_at) values (?, ?, ?, ?, ?)",
            [
                (str(uuid4()), search_run_id, market.market_id, market.model_dump_json(), now_iso())
                for market in markets
            ],
        )
        self.db.commit()


class StructuredMarketRepository:
    def __init__(self, db: sqlite3.Connection) -> None:
        self.db = db

    def save_many(self, search_run_id: str, markets: list[StructuredMarket]) -> None:
        self.db.executemany(
            "insert into structured_markets (id, search_run_id, market_id, payload_json, created_at) values (?, ?, ?, ?, ?)",
            [
                (str(uuid4()), search_run_id, market.market_id, market.model_dump_json(), now_iso())
                for market in markets
            ],
        )
        self.db.commit()

    def list_for_run(self, search_run_id: str) -> list[StructuredMarket]:
        rows = self.db.execute(
            "select payload_json from structured_markets where search_run_id = ? order by created_at asc",
            (search_run_id,),
        ).fetchall()
        return [StructuredMarket.model_validate_json(row[0]) for row in rows]

    def save_review(self, search_run_id: str, markets: list[StructuredMarket]) -> str:
        review_id = str(uuid4())
        payload = json.dumps([market.model_dump(mode="json") for market in markets])
        self.db.execute(
            "insert into structured_market_reviews (id, search_run_id, payload_json, created_at) values (?, ?, ?, ?)",
            (review_id, search_run_id, payload, now_iso()),
        )
        self.db.commit()
        return review_id

    def list_reviewed_for_run(self, search_run_id: str) -> list[StructuredMarket]:
        row = self.db.execute(
            "select payload_json from structured_market_reviews where search_run_id = ? order by created_at desc limit 1",
            (search_run_id,),
        ).fetchone()
        if row is None:
            return []
        payload = json.loads(row[0])
        return [StructuredMarket.model_validate(item) for item in payload]

    def list_effective_for_run(self, search_run_id: str) -> list[StructuredMarket]:
        reviewed = self.list_reviewed_for_run(search_run_id)
        if reviewed:
            return reviewed
        return self.list_for_run(search_run_id)


class AggregationRunRepository:
    def __init__(self, db: sqlite3.Connection) -> None:
        self.db = db

    def create(
        self, search_run_id: str, payload: AggregationInput, result: AggregationResult
    ) -> str:
        run_id = str(uuid4())
        self.db.execute(
            "insert into aggregation_runs (id, search_run_id, object_name, input_json, result_json, created_at) values (?, ?, ?, ?, ?, ?)",
            (
                run_id,
                search_run_id,
                payload.object_name,
                payload.model_dump_json(),
                result.model_dump_json(),
                now_iso(),
            ),
        )
        self.db.commit()
        return run_id
