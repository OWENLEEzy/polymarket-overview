# Polymarket Overview Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local Python web tool that searches Polymarket on demand, uses AI to structure related markets, lets the user confirm fields, and computes deterministic macro overviews.

**Architecture:** FastAPI serves a simple Jinja2 + HTMX UI. A typed service layer separates Polymarket API reads, AI structuring, recall ranking, SQLite audit storage, and deterministic aggregation. AI outputs pass through Pydantic before storage or analysis.

**Tech Stack:** Python 3.12, FastAPI, Jinja2, HTMX, httpx, Pydantic, SQLite, rank-bm25, sentence-transformers, pandas, numpy, scikit-learn, Plotly, pytest.

---

## 0. File Structure

Create this structure under `/Users/owenlee/Desktop/polymarket-overview`:

```text
polymarket-overview/
  pyproject.toml
  README.md
  .env.example
  .gitignore
  app/
    __init__.py
    main.py
    config.py
    models.py
    ai/
      __init__.py
      provider.py
      deepseek.py
      prompts.py
    analysis/
      __init__.py
      aggregators.py
      anomalies.py
    polymarket/
      __init__.py
      client.py
      normalizer.py
    recall/
      __init__.py
      ranker.py
    storage/
      __init__.py
      db.py
      repositories.py
      schema.sql
    web/
      __init__.py
      routes.py
      templates/
        base.html
        search.html
        objects.html
        review.html
        overview.html
      static/
        app.css
  tests/
    test_models.py
    test_polymarket_normalizer.py
    test_ai_provider.py
    test_recall_ranker.py
    test_aggregators.py
    test_storage.py
    test_web_routes.py
```

Responsibility boundaries:

- `app/models.py`: all Pydantic models and enums used across layers.
- `app/polymarket/*`: public Polymarket reads and raw payload normalization.
- `app/ai/*`: provider interface, DeepSeek call wrapper, and prompt text.
- `app/recall/*`: local BM25 and embedding-assisted ranking.
- `app/analysis/*`: deterministic aggregation and anomaly detection.
- `app/storage/*`: SQLite schema and repositories.
- `app/web/*`: routes, forms, and templates.

## 1. Scaffold Python Project

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `.env.example`
- Create: `README.md`
- Create: `app/__init__.py`
- Create: `app/config.py`
- Test: `tests/test_models.py`

- [ ] **Step 1: Write project metadata**

Create `pyproject.toml`:

```toml
[project]
name = "polymarket-overview"
version = "0.1.0"
description = "Local Polymarket macro overview tool"
requires-python = ">=3.12"
dependencies = [
  "fastapi>=0.115.0",
  "uvicorn[standard]>=0.30.0",
  "jinja2>=3.1.4",
  "python-multipart>=0.0.9",
  "httpx>=0.27.0",
  "pydantic>=2.8.0",
  "pydantic-settings>=2.4.0",
  "rank-bm25>=0.2.2",
  "sentence-transformers>=3.0.0",
  "pandas>=2.2.0",
  "numpy>=1.26.0",
  "scikit-learn>=1.5.0",
  "plotly>=5.22.0",
]

[project.optional-dependencies]
dev = [
  "pytest>=8.2.0",
  "pytest-asyncio>=0.23.0",
  "ruff>=0.6.0",
  "mypy>=1.10.0",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"

[tool.ruff]
line-length = 100

[tool.mypy]
python_version = "3.12"
strict = true
```

- [ ] **Step 2: Add local ignores and env example**

Create `.gitignore`:

```gitignore
.env
.venv/
__pycache__/
.pytest_cache/
.mypy_cache/
.ruff_cache/
data/
*.sqlite
```

Create `.env.example`:

```text
POLYMARKET_OVERVIEW_DB_PATH=data/polymarket-overview.sqlite
POLYMARKET_OVERVIEW_HOST=127.0.0.1
POLYMARKET_OVERVIEW_PORT=8787
POLYMARKET_OVERVIEW_AI_PROVIDER=deepseek
DEEPSEEK_API_KEY=
DEEPSEEK_MODEL=deepseek-chat
```

- [ ] **Step 3: Write config loader**

Create `app/__init__.py` as an empty file.

Create `app/config.py`:

```python
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="POLYMARKET_OVERVIEW_")

    db_path: str = "data/polymarket-overview.sqlite"
    host: str = "127.0.0.1"
    port: int = 8787
    ai_provider: str = "deepseek"
    deepseek_api_key: str | None = None
    deepseek_model: str = "deepseek-chat"


@lru_cache
def get_settings() -> Settings:
    return Settings()
```

- [ ] **Step 4: Write smoke README**

Create `README.md`:

```markdown
# Polymarket Overview

Local web tool for turning related Polymarket markets into macro overviews.

## Local Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
pytest
uvicorn app.main:app --host 127.0.0.1 --port 8787
```

Open `http://127.0.0.1:8787`.
```

- [ ] **Step 5: Run bootstrap checks**

Run:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Expected: dependency install succeeds and pytest reports no collected tests or all collected tests passing.

- [ ] **Step 6: Commit scaffold**

```bash
git add pyproject.toml .gitignore .env.example README.md app/__init__.py app/config.py
git commit -m "chore: scaffold polymarket overview project"
```

## 2. Define Core Models

**Files:**
- Create: `app/models.py`
- Test: `tests/test_models.py`

- [ ] **Step 1: Write failing model tests**

Create `tests/test_models.py`:

```python
from app.models import (
    AggregationInput,
    ObjectType,
    Operator,
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

    payload = AggregationInput(object_name="X", object_type=ObjectType.CONTINUOUS, markets=[included, excluded])

    assert [market.market_id for market in payload.included_markets()] == ["m1"]
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
pytest tests/test_models.py -v
```

Expected: FAIL because `app.models` does not exist.

- [ ] **Step 3: Implement models**

Create `app/models.py`:

```python
from enum import StrEnum
from typing import Any
from pydantic import BaseModel, Field, field_validator


class ObjectType(StrEnum):
    CONTINUOUS = "continuous"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    TIME = "time"


class Operator(StrEnum):
    GREATER_THAN = ">"
    GREATER_THAN_OR_EQUAL = ">="
    LESS_THAN = "<"
    LESS_THAN_OR_EQUAL = "<="
    EQUAL = "="
    RANGE = "range"
    CATEGORY = "category"


class RawMarket(BaseModel):
    market_id: str
    event_id: str | None = None
    question: str
    event_title: str | None = None
    outcomes: list[str] = Field(default_factory=list)
    outcome_prices: list[float] = Field(default_factory=list)
    token_ids: list[str] = Field(default_factory=list)
    closed: bool = False
    archived: bool = False
    liquidity: float | None = None
    volume: float | None = None
    raw_json: dict[str, Any]


class StructuredMarket(BaseModel):
    market_id: str
    question: str
    event_title: str
    object_name: str
    object_type: ObjectType
    operator: Operator
    threshold_value: float | None
    threshold_unit: str | None
    category_value: str | None
    probability: float = Field(ge=0.0, le=1.0)
    probability_source: str
    resolution_date: str | None
    include: bool
    confidence: float = Field(ge=0.0, le=1.0)
    explanation: str

    @field_validator("threshold_value")
    @classmethod
    def threshold_required_for_threshold_ops(cls, value: float | None, info: Any) -> float | None:
        operator = info.data.get("operator")
        if operator in {Operator.GREATER_THAN, Operator.GREATER_THAN_OR_EQUAL, Operator.LESS_THAN, Operator.LESS_THAN_OR_EQUAL} and value is None:
            raise ValueError("threshold_value is required for threshold operators")
        return value


class DataObjectCandidate(BaseModel):
    object_name: str
    object_type: ObjectType
    market_ids: list[str]
    confidence: float = Field(ge=0.0, le=1.0)
    explanation: str


class StructuredExtraction(BaseModel):
    topic: str
    objects: list[DataObjectCandidate]
    markets: list[StructuredMarket]


class AggregationInput(BaseModel):
    object_name: str
    object_type: ObjectType
    markets: list[StructuredMarket]

    def included_markets(self) -> list[StructuredMarket]:
        return [market for market in self.markets if market.include]


class IntervalProbability(BaseModel):
    label: str
    lower: float | None
    upper: float | None
    probability: float
    corrected: bool = False


class AggregationResult(BaseModel):
    object_name: str
    object_type: ObjectType
    rows: list[dict[str, Any]]
    chart_json: dict[str, Any]
    anomalies: list[str]
```

- [ ] **Step 4: Run model tests**

Run:

```bash
pytest tests/test_models.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit models**

```bash
git add app/models.py tests/test_models.py
git commit -m "feat: add core market models"
```

## 3. Add SQLite Storage And Audit Tables

**Files:**
- Create: `app/storage/__init__.py`
- Create: `app/storage/schema.sql`
- Create: `app/storage/db.py`
- Create: `app/storage/repositories.py`
- Test: `tests/test_storage.py`

- [ ] **Step 1: Write failing storage tests**

Create `tests/test_storage.py`:

```python
import sqlite3
from app.storage.db import migrate
from app.storage.repositories import SearchRunRepository


def test_migrate_creates_audit_tables() -> None:
    db = sqlite3.connect(":memory:")
    migrate(db)

    rows = db.execute("select name from sqlite_master where type = 'table' order by name").fetchall()
    names = [row[0] for row in rows]

    assert "search_runs" in names
    assert "raw_markets" in names
    assert "structured_markets" in names
    assert "aggregation_runs" in names


def test_search_run_repository_creates_run() -> None:
    db = sqlite3.connect(":memory:")
    migrate(db)

    repo = SearchRunRepository(db)
    run_id = repo.create(topic="Anthropic IPO", provider="deepseek", search_terms=["Anthropic IPO"])

    row = db.execute("select topic, provider from search_runs where id = ?", (run_id,)).fetchone()
    assert row == ("Anthropic IPO", "deepseek")
```

- [ ] **Step 2: Run storage tests to verify failure**

Run:

```bash
pytest tests/test_storage.py -v
```

Expected: FAIL because storage modules do not exist.

- [ ] **Step 3: Implement schema and repositories**

Create `app/storage/__init__.py` as an empty file.

Create `app/storage/schema.sql`:

```sql
create table if not exists search_runs (
  id text primary key,
  topic text not null,
  provider text not null,
  search_terms_json text not null,
  created_at text not null
);

create table if not exists raw_markets (
  id text primary key,
  search_run_id text not null references search_runs(id) on delete cascade,
  market_id text not null,
  payload_json text not null,
  created_at text not null
);

create table if not exists structured_markets (
  id text primary key,
  search_run_id text not null references search_runs(id) on delete cascade,
  market_id text not null,
  payload_json text not null,
  created_at text not null
);

create table if not exists aggregation_runs (
  id text primary key,
  search_run_id text not null references search_runs(id) on delete cascade,
  object_name text not null,
  input_json text not null,
  result_json text not null,
  created_at text not null
);

create index if not exists idx_raw_markets_search_run_id on raw_markets(search_run_id);
create index if not exists idx_structured_markets_search_run_id on structured_markets(search_run_id);
create index if not exists idx_aggregation_runs_search_run_id on aggregation_runs(search_run_id);
```

Create `app/storage/db.py`:

```python
import sqlite3
from pathlib import Path


def connect(db_path: str) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(db_path)
    db.execute("pragma foreign_keys = on")
    return db


def migrate(db: sqlite3.Connection) -> None:
    schema_path = Path(__file__).with_name("schema.sql")
    db.executescript(schema_path.read_text())
    db.commit()
```

Create `app/storage/repositories.py`:

```python
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


class AggregationRunRepository:
    def __init__(self, db: sqlite3.Connection) -> None:
        self.db = db

    def create(self, search_run_id: str, payload: AggregationInput, result: AggregationResult) -> str:
        run_id = str(uuid4())
        self.db.execute(
            "insert into aggregation_runs (id, search_run_id, object_name, input_json, result_json, created_at) values (?, ?, ?, ?, ?, ?)",
            (run_id, search_run_id, payload.object_name, payload.model_dump_json(), result.model_dump_json(), now_iso()),
        )
        self.db.commit()
        return run_id
```

- [ ] **Step 4: Run storage tests**

Run:

```bash
pytest tests/test_storage.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit storage layer**

```bash
git add app/storage tests/test_storage.py
git commit -m "feat: add sqlite audit storage"
```

## 4. Normalize Polymarket Markets

**Files:**
- Create: `app/polymarket/__init__.py`
- Create: `app/polymarket/normalizer.py`
- Create: `app/polymarket/client.py`
- Test: `tests/test_polymarket_normalizer.py`

- [ ] **Step 1: Write failing normalizer tests**

Create `tests/test_polymarket_normalizer.py`:

```python
from app.polymarket.normalizer import normalize_market


def test_normalize_gamma_market_parses_outcomes_prices_and_tokens() -> None:
    payload = {
        "id": "123",
        "conditionId": "cond-1",
        "question": "Will Anthropic IPO above $100B?",
        "outcomes": '["Yes","No"]',
        "outcomePrices": '["0.42","0.58"]',
        "clobTokenIds": '["token-yes","token-no"]',
        "closed": False,
        "archived": False,
        "liquidity": "1020.5",
        "volume": "991.2",
        "events": [{"id": "e1", "title": "Anthropic IPO"}],
    }

    market = normalize_market(payload)

    assert market.market_id == "123"
    assert market.event_id == "e1"
    assert market.outcomes == ["Yes", "No"]
    assert market.outcome_prices == [0.42, 0.58]
    assert market.token_ids == ["token-yes", "token-no"]
```

- [ ] **Step 2: Run normalizer test to verify failure**

Run:

```bash
pytest tests/test_polymarket_normalizer.py -v
```

Expected: FAIL because normalizer does not exist.

- [ ] **Step 3: Implement normalizer and client**

Create `app/polymarket/__init__.py` as an empty file.

Create `app/polymarket/normalizer.py`:

```python
import json
from typing import Any
from app.models import RawMarket


def parse_json_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        parsed = json.loads(value)
        return parsed if isinstance(parsed, list) else []
    return []


def parse_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def normalize_market(payload: dict[str, Any]) -> RawMarket:
    events = payload.get("events")
    event = events[0] if isinstance(events, list) and events else {}
    outcomes = [str(item) for item in parse_json_list(payload.get("outcomes"))]
    prices = [float(item) for item in parse_json_list(payload.get("outcomePrices"))]
    tokens = [str(item) for item in parse_json_list(payload.get("clobTokenIds"))]

    return RawMarket(
        market_id=str(payload.get("id") or payload.get("conditionId")),
        event_id=str(event.get("id")) if event.get("id") is not None else None,
        question=str(payload.get("question") or ""),
        event_title=str(event.get("title")) if event.get("title") is not None else None,
        outcomes=outcomes,
        outcome_prices=prices,
        token_ids=tokens,
        closed=bool(payload.get("closed", False)),
        archived=bool(payload.get("archived", False)),
        liquidity=parse_float(payload.get("liquidity")),
        volume=parse_float(payload.get("volume")),
        raw_json=payload,
    )
```

Create `app/polymarket/client.py`:

```python
from typing import Any
import httpx
from app.models import RawMarket
from app.polymarket.normalizer import normalize_market


GAMMA_BASE_URL = "https://gamma-api.polymarket.com"
CLOB_BASE_URL = "https://clob.polymarket.com"


class PolymarketClient:
    def __init__(self, http_client: httpx.AsyncClient | None = None) -> None:
        self.http_client = http_client or httpx.AsyncClient(timeout=20.0)

    async def search_markets(self, query: str, limit: int = 50) -> list[RawMarket]:
        response = await self.http_client.get(
            f"{GAMMA_BASE_URL}/markets",
            params={"search": query, "limit": limit, "active": "true"},
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list):
            return []
        return [normalize_market(item) for item in payload if isinstance(item, dict)]

    async def get_midpoints(self, token_ids: list[str]) -> dict[str, float]:
        if not token_ids:
            return {}
        response = await self.http_client.get(
            f"{CLOB_BASE_URL}/midpoints",
            params={"token_ids": ",".join(token_ids)},
        )
        response.raise_for_status()
        payload: dict[str, Any] = response.json()
        return {token_id: float(value) for token_id, value in payload.items()}
```

- [ ] **Step 4: Run normalizer test**

Run:

```bash
pytest tests/test_polymarket_normalizer.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit Polymarket layer**

```bash
git add app/polymarket tests/test_polymarket_normalizer.py
git commit -m "feat: add polymarket market normalization"
```

## 5. Add AI Provider Interface And DeepSeek Demo Provider

**Files:**
- Create: `app/ai/__init__.py`
- Create: `app/ai/provider.py`
- Create: `app/ai/deepseek.py`
- Create: `app/ai/prompts.py`
- Test: `tests/test_ai_provider.py`

- [ ] **Step 1: Write failing AI provider tests**

Create `tests/test_ai_provider.py`:

```python
import pytest
from app.ai.provider import parse_structured_extraction
from app.models import ObjectType


def test_parse_structured_extraction_validates_json() -> None:
    payload = {
        "topic": "Anthropic IPO",
        "objects": [
            {
                "object_name": "Anthropic IPO valuation",
                "object_type": "continuous",
                "market_ids": ["m1"],
                "confidence": 0.9,
                "explanation": "Valuation threshold markets.",
            }
        ],
        "markets": [
            {
                "market_id": "m1",
                "question": "Will Anthropic IPO above $100B?",
                "event_title": "Anthropic IPO",
                "object_name": "Anthropic IPO valuation",
                "object_type": "continuous",
                "operator": ">",
                "threshold_value": 100.0,
                "threshold_unit": "USD billion",
                "category_value": None,
                "probability": 0.42,
                "probability_source": "outcome_price",
                "resolution_date": "2027-12-31",
                "include": True,
                "confidence": 0.87,
                "explanation": "Yes outcome maps to P(X > 100B).",
            }
        ],
    }

    extraction = parse_structured_extraction(payload)

    assert extraction.objects[0].object_type == ObjectType.CONTINUOUS
    assert extraction.markets[0].threshold_value == 100.0


def test_parse_structured_extraction_rejects_invalid_probability() -> None:
    payload = {
        "topic": "X",
        "objects": [],
        "markets": [
            {
                "market_id": "m1",
                "question": "X?",
                "event_title": "X",
                "object_name": "X",
                "object_type": "boolean",
                "operator": "=",
                "threshold_value": None,
                "threshold_unit": None,
                "category_value": "yes",
                "probability": 1.2,
                "probability_source": "outcome_price",
                "resolution_date": None,
                "include": True,
                "confidence": 0.5,
                "explanation": "Invalid probability.",
            }
        ],
    }

    with pytest.raises(ValueError):
        parse_structured_extraction(payload)
```

- [ ] **Step 2: Run AI tests to verify failure**

Run:

```bash
pytest tests/test_ai_provider.py -v
```

Expected: FAIL because `app.ai.provider` does not exist.

- [ ] **Step 3: Implement AI provider interface**

Create `app/ai/__init__.py` as an empty file.

Create `app/ai/provider.py`:

```python
from abc import ABC, abstractmethod
from typing import Any
from app.models import RawMarket, StructuredExtraction


class AIProvider(ABC):
    @abstractmethod
    async def plan_search_terms(self, topic: str) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    async def structure_markets(self, topic: str, markets: list[RawMarket]) -> StructuredExtraction:
        raise NotImplementedError


def parse_structured_extraction(payload: dict[str, Any]) -> StructuredExtraction:
    return StructuredExtraction.model_validate(payload)
```

Create `app/ai/prompts.py`:

```python
SEARCH_PLANNING_PROMPT = """Return 3 to 6 Polymarket search queries for the topic.
Use related aliases and event phrasing. Return JSON: {"queries": ["..."]}.
"""

STRUCTURE_MARKETS_PROMPT = """Structure Polymarket markets into data objects.
Use only the market data provided. Return strict JSON matching:
{
  "topic": string,
  "objects": [
    {"object_name": string, "object_type": "continuous|categorical|boolean|time", "market_ids": [string], "confidence": number, "explanation": string}
  ],
  "markets": [
    {
      "market_id": string,
      "question": string,
      "event_title": string,
      "object_name": string,
      "object_type": "continuous|categorical|boolean|time",
      "operator": ">|>=|<|<=|=|range|category",
      "threshold_value": number|null,
      "threshold_unit": string|null,
      "category_value": string|null,
      "probability": number,
      "probability_source": string,
      "resolution_date": string|null,
      "include": boolean,
      "confidence": number,
      "explanation": string
    }
  ]
}
"""
```

Create `app/ai/deepseek.py`:

```python
import json
from typing import Any
import httpx
from app.ai.provider import AIProvider, parse_structured_extraction
from app.ai.prompts import SEARCH_PLANNING_PROMPT, STRUCTURE_MARKETS_PROMPT
from app.models import RawMarket, StructuredExtraction


class DeepSeekProvider(AIProvider):
    def __init__(self, api_key: str, model: str = "deepseek-chat", http_client: httpx.AsyncClient | None = None) -> None:
        self.api_key = api_key
        self.model = model
        self.http_client = http_client or httpx.AsyncClient(timeout=60.0)

    async def plan_search_terms(self, topic: str) -> list[str]:
        payload = await self._chat_json(SEARCH_PLANNING_PROMPT, topic)
        queries = payload.get("queries", [])
        return [str(item) for item in queries if str(item).strip()]

    async def structure_markets(self, topic: str, markets: list[RawMarket]) -> StructuredExtraction:
        market_payload = [market.model_dump() for market in markets]
        payload = await self._chat_json(STRUCTURE_MARKETS_PROMPT, json.dumps({"topic": topic, "markets": market_payload}))
        return parse_structured_extraction(payload)

    async def _chat_json(self, system_prompt: str, user_content: str) -> dict[str, Any]:
        response = await self.http_client.post(
            "https://api.deepseek.com/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                "response_format": {"type": "json_object"},
            },
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        return json.loads(content)
```

- [ ] **Step 4: Run AI provider tests**

Run:

```bash
pytest tests/test_ai_provider.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit AI provider**

```bash
git add app/ai tests/test_ai_provider.py
git commit -m "feat: add ai provider interface"
```

## 6. Add Recall Ranking

**Files:**
- Create: `app/recall/__init__.py`
- Create: `app/recall/ranker.py`
- Test: `tests/test_recall_ranker.py`

- [ ] **Step 1: Write failing recall tests**

Create `tests/test_recall_ranker.py`:

```python
from app.models import RawMarket
from app.recall.ranker import rank_markets


def make_market(market_id: str, question: str) -> RawMarket:
    return RawMarket(
        market_id=market_id,
        question=question,
        event_title="Test",
        outcomes=["Yes", "No"],
        outcome_prices=[0.5, 0.5],
        raw_json={"id": market_id, "question": question},
    )


def test_rank_markets_prioritizes_query_terms() -> None:
    markets = [
        make_market("m1", "Will Anthropic IPO above $100B?"),
        make_market("m2", "Will Bitcoin hit $100k?"),
    ]

    ranked = rank_markets("Anthropic IPO valuation", markets)

    assert ranked[0].market.market_id == "m1"
    assert ranked[0].score > ranked[1].score
```

- [ ] **Step 2: Run recall tests to verify failure**

Run:

```bash
pytest tests/test_recall_ranker.py -v
```

Expected: FAIL because recall ranker does not exist.

- [ ] **Step 3: Implement BM25 ranker**

Create `app/recall/__init__.py` as an empty file.

Create `app/recall/ranker.py`:

```python
import re
from dataclasses import dataclass
from rank_bm25 import BM25Okapi
from app.models import RawMarket


TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


@dataclass(frozen=True)
class RankedMarket:
    market: RawMarket
    score: float


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_RE.finditer(text)]


def market_text(market: RawMarket) -> str:
    return " ".join(
        part
        for part in [
            market.question,
            market.event_title or "",
            " ".join(market.outcomes),
        ]
        if part
    )


def rank_markets(query: str, markets: list[RawMarket]) -> list[RankedMarket]:
    if not markets:
        return []
    corpus = [tokenize(market_text(market)) for market in markets]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(tokenize(query))
    ranked = [RankedMarket(market=market, score=float(score)) for market, score in zip(markets, scores, strict=True)]
    return sorted(ranked, key=lambda item: item.score, reverse=True)
```

- [ ] **Step 4: Run recall tests**

Run:

```bash
pytest tests/test_recall_ranker.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit recall layer**

```bash
git add app/recall tests/test_recall_ranker.py
git commit -m "feat: add market recall ranking"
```

## 7. Add Deterministic Aggregators

**Files:**
- Create: `app/analysis/__init__.py`
- Create: `app/analysis/aggregators.py`
- Create: `app/analysis/anomalies.py`
- Test: `tests/test_aggregators.py`

- [ ] **Step 1: Write failing aggregation tests**

Create `tests/test_aggregators.py`:

```python
from app.analysis.aggregators import aggregate
from app.models import AggregationInput, ObjectType, Operator, StructuredMarket


def threshold(market_id: str, value: float, probability: float) -> StructuredMarket:
    return StructuredMarket(
        market_id=market_id,
        question=f"Will X be greater than {value}?",
        event_title="X",
        object_name="X",
        object_type=ObjectType.CONTINUOUS,
        operator=Operator.GREATER_THAN,
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
```

- [ ] **Step 2: Run aggregation tests to verify failure**

Run:

```bash
pytest tests/test_aggregators.py -v
```

Expected: FAIL because analysis aggregators do not exist.

- [ ] **Step 3: Implement aggregators**

Create `app/analysis/__init__.py` as an empty file.

Create `app/analysis/anomalies.py`:

```python
from app.models import StructuredMarket


def has_survival_monotonicity_conflict(markets: list[StructuredMarket]) -> bool:
    points = sorted(
        [
            (market.threshold_value, market.probability)
            for market in markets
            if market.threshold_value is not None and market.operator in {">", ">="}
        ],
        key=lambda item: item[0],
    )
    return any(next_probability > probability for (_, probability), (_, next_probability) in zip(points, points[1:], strict=False))
```

Create `app/analysis/aggregators.py`:

```python
from app.analysis.anomalies import has_survival_monotonicity_conflict
from app.models import AggregationInput, AggregationResult, ObjectType, StructuredMarket


def aggregate(payload: AggregationInput) -> AggregationResult:
    if payload.object_type in {ObjectType.CONTINUOUS, ObjectType.TIME}:
        return aggregate_continuous(payload)
    if payload.object_type in {ObjectType.CATEGORICAL, ObjectType.BOOLEAN}:
        return aggregate_categorical(payload)
    raise ValueError(f"Unsupported object type: {payload.object_type}")


def aggregate_continuous(payload: AggregationInput) -> AggregationResult:
    markets = sorted(
        [market for market in payload.included_markets() if market.threshold_value is not None],
        key=lambda market: market.threshold_value or 0.0,
    )
    anomalies: list[str] = []
    if has_survival_monotonicity_conflict(markets):
        anomalies.append("monotonicity_conflict")

    rows: list[dict[str, float | str | None]] = []
    previous_threshold: float | None = None
    previous_survival = 1.0

    for market in markets:
        survival = market.probability
        interval_probability = max(previous_survival - survival, 0.0)
        label = (
            f"X <= {market.threshold_value:g} {market.threshold_unit}"
            if previous_threshold is None
            else f"{previous_threshold:g} < X <= {market.threshold_value:g} {market.threshold_unit}"
        )
        rows.append(
            {
                "label": label,
                "lower": previous_threshold,
                "upper": market.threshold_value,
                "probability": interval_probability,
            }
        )
        previous_threshold = market.threshold_value
        previous_survival = survival

    if markets:
        last = markets[-1]
        rows.append(
            {
                "label": f"X > {last.threshold_value:g} {last.threshold_unit}",
                "lower": last.threshold_value,
                "upper": None,
                "probability": last.probability,
            }
        )

    return AggregationResult(
        object_name=payload.object_name,
        object_type=payload.object_type,
        rows=rows,
        chart_json={"type": "bar", "labels": [row["label"] for row in rows], "values": [row["probability"] for row in rows]},
        anomalies=anomalies,
    )


def aggregate_categorical(payload: AggregationInput) -> AggregationResult:
    rows = [
        {
            "category": market.category_value or market.question,
            "probability": market.probability,
            "market_id": market.market_id,
        }
        for market in payload.included_markets()
    ]
    return AggregationResult(
        object_name=payload.object_name,
        object_type=payload.object_type,
        rows=rows,
        chart_json={"type": "bar", "labels": [row["category"] for row in rows], "values": [row["probability"] for row in rows]},
        anomalies=[],
    )
```

- [ ] **Step 4: Run aggregation tests**

Run:

```bash
pytest tests/test_aggregators.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit aggregation layer**

```bash
git add app/analysis tests/test_aggregators.py
git commit -m "feat: add deterministic aggregators"
```

## 8. Add FastAPI Web Routes And Templates

**Files:**
- Create: `app/main.py`
- Create: `app/web/__init__.py`
- Create: `app/web/routes.py`
- Create: `app/web/templates/base.html`
- Create: `app/web/templates/search.html`
- Create: `app/web/templates/objects.html`
- Create: `app/web/templates/review.html`
- Create: `app/web/templates/overview.html`
- Create: `app/web/static/app.css`
- Test: `tests/test_web_routes.py`

- [ ] **Step 1: Write failing route tests**

Create `tests/test_web_routes.py`:

```python
from fastapi.testclient import TestClient
from app.main import app


def test_search_page_loads() -> None:
    client = TestClient(app)

    response = client.get("/")

    assert response.status_code == 200
    assert "Polymarket Overview" in response.text
    assert "name=\"topic\"" in response.text
```

- [ ] **Step 2: Run route test to verify failure**

Run:

```bash
pytest tests/test_web_routes.py -v
```

Expected: FAIL because `app.main` does not exist.

- [ ] **Step 3: Implement app and search page**

Create `app/web/__init__.py` as an empty file.

Create `app/main.py`:

```python
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.web.routes import router


app = FastAPI(title="Polymarket Overview")
app.mount("/static", StaticFiles(directory="app/web/static"), name="static")
app.include_router(router)
```

Create `app/web/routes.py`:

```python
from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates


router = APIRouter()
templates = Jinja2Templates(directory="app/web/templates")


@router.get("/")
async def search_page(request: Request):
    return templates.TemplateResponse("search.html", {"request": request, "title": "Polymarket Overview"})
```

Create `app/web/templates/base.html`:

```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{{ title }}</title>
    <link rel="stylesheet" href="/static/app.css">
    <script src="https://unpkg.com/htmx.org@2.0.4"></script>
  </head>
  <body>
    <main>
      {% block content %}{% endblock %}
    </main>
  </body>
</html>
```

Create `app/web/templates/search.html`:

```html
{% extends "base.html" %}
{% block content %}
<section class="panel">
  <h1>Polymarket Overview</h1>
  <form method="post" action="/search">
    <label for="topic">Topic</label>
    <input id="topic" name="topic" value="" aria-label="Topic" required>
    <button type="submit">Search</button>
  </form>
</section>
{% endblock %}
```

Create simple page templates required by future routes:

`app/web/templates/objects.html`:

```html
{% extends "base.html" %}
{% block content %}<h1>Objects</h1>{% endblock %}
```

`app/web/templates/review.html`:

```html
{% extends "base.html" %}
{% block content %}<h1>Review</h1>{% endblock %}
```

`app/web/templates/overview.html`:

```html
{% extends "base.html" %}
{% block content %}<h1>Overview</h1>{% endblock %}
```

Create `app/web/static/app.css`:

```css
body {
  margin: 0;
  font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background: #f7f7f4;
  color: #202124;
}

main {
  max-width: 1080px;
  margin: 0 auto;
  padding: 32px;
}

.panel {
  background: #fff;
  border: 1px solid #deded8;
  border-radius: 8px;
  padding: 24px;
}

input,
button {
  font: inherit;
  padding: 10px 12px;
}

input {
  min-width: 320px;
}
```

- [ ] **Step 4: Run route tests**

Run:

```bash
pytest tests/test_web_routes.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit web shell**

```bash
git add app/main.py app/web tests/test_web_routes.py
git commit -m "feat: add local web shell"
```

## 9. Wire Search Flow End To End

**Files:**
- Modify: `app/web/routes.py`
- Modify: `app/web/templates/objects.html`
- Modify: `app/web/templates/review.html`
- Modify: `app/web/templates/overview.html`
- Test: `tests/test_web_routes.py`

- [ ] **Step 1: Add route test with mocked service behavior**

Append to `tests/test_web_routes.py`:

```python
def test_search_post_returns_objects_page() -> None:
    client = TestClient(app)

    response = client.post("/search", data={"topic": "Anthropic IPO"})

    assert response.status_code == 200
    assert "Data Objects" in response.text
    assert "Anthropic IPO" in response.text
```

- [ ] **Step 2: Run route test to verify failure**

Run:

```bash
pytest tests/test_web_routes.py::test_search_post_returns_objects_page -v
```

Expected: FAIL because `/search` is not registered.

- [ ] **Step 3: Add deterministic demo route for UI wiring**

Modify `app/web/routes.py`:

```python
from fastapi import APIRouter, Form, Request
from fastapi.templating import Jinja2Templates


router = APIRouter()
templates = Jinja2Templates(directory="app/web/templates")


@router.get("/")
async def search_page(request: Request):
    return templates.TemplateResponse("search.html", {"request": request, "title": "Polymarket Overview"})


@router.post("/search")
async def search_submit(request: Request, topic: str = Form(...)):
    objects = [
        {"object_name": f"{topic} occurrence", "object_type": "boolean", "confidence": 0.75},
        {"object_name": f"{topic} valuation", "object_type": "continuous", "confidence": 0.72},
    ]
    return templates.TemplateResponse(
        "objects.html",
        {"request": request, "title": "Data Objects", "topic": topic, "objects": objects},
    )
```

Modify `app/web/templates/objects.html`:

```html
{% extends "base.html" %}
{% block content %}
<section class="panel">
  <h1>Data Objects</h1>
  <p>{{ topic }}</p>
  <table>
    <thead>
      <tr><th>Object</th><th>Type</th><th>Confidence</th></tr>
    </thead>
    <tbody>
      {% for object in objects %}
      <tr>
        <td>{{ object.object_name }}</td>
        <td>{{ object.object_type }}</td>
        <td>{{ "%.2f"|format(object.confidence) }}</td>
      </tr>
      {% endfor %}
    </tbody>
  </table>
</section>
{% endblock %}
```

- [ ] **Step 4: Run web tests**

Run:

```bash
pytest tests/test_web_routes.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit route wiring**

```bash
git add app/web/routes.py app/web/templates/objects.html tests/test_web_routes.py
git commit -m "feat: wire topic search page flow"
```

## 10. Final Verification

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Run full test suite**

Run:

```bash
pytest -v
```

Expected: all tests PASS.

- [ ] **Step 2: Run lint and type checks**

Run:

```bash
ruff check app tests
mypy app
```

Expected: both commands PASS.

- [ ] **Step 3: Start local server**

Run:

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8787
```

Expected: server starts at `http://127.0.0.1:8787`.

- [ ] **Step 4: Open browser and verify UI**

Open:

```text
http://127.0.0.1:8787
```

Expected:

- Search page loads.
- Topic form accepts `Anthropic IPO`.
- Submitting shows the data objects page.

- [ ] **Step 5: Commit verification docs**

Update `README.md` with the verified local URL and commands:

```markdown
## Verified Commands

```bash
pytest -v
ruff check app tests
mypy app
uvicorn app.main:app --host 127.0.0.1 --port 8787
```
```

Commit:

```bash
git add README.md
git commit -m "docs: add verified local run commands"
```

## Self-Review Checklist

- Spec coverage: tasks cover project scaffold, data models, audit storage, Polymarket normalization, AI provider abstraction, recall ranking, deterministic aggregation, local web UI, and verification.
- Scope check: the plan does not include full Polymarket indexing, remote deployment, trading, paid data, multi-user auth, or T46 integration.
- Type consistency: `ObjectType`, `Operator`, `RawMarket`, `StructuredMarket`, `StructuredExtraction`, `AggregationInput`, and `AggregationResult` are defined in Task 2 and used consistently afterward.
- AI boundary: DeepSeek only returns search terms and structured JSON; aggregation is implemented in `app/analysis`.
- Audit boundary: raw markets, structured markets, search runs, and aggregation runs have dedicated SQLite tables.
