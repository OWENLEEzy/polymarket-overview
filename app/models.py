from enum import StrEnum
import hashlib
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


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
    slug: str | None = None
    end_date: str | None = None
    group_item_title: str | None = None
    group_item_range: list[str] = Field(default_factory=list)
    outcomes: list[str] = Field(default_factory=list)
    outcome_prices: list[float] = Field(default_factory=list)
    token_ids: list[str] = Field(default_factory=list)
    closed: bool = False
    archived: bool = False
    active: bool = True
    liquidity: float | None = None
    volume: float | None = None
    raw_json: dict[str, Any]

    def display_title(self) -> str:
        if self.event_title and self.event_title != self.question:
            return f"{self.event_title} - {self.question}"
        return self.question


class StructuredMarket(BaseModel):
    object_id: str | None = None
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
        threshold_ops = {
            Operator.GREATER_THAN,
            Operator.GREATER_THAN_OR_EQUAL,
            Operator.LESS_THAN,
            Operator.LESS_THAN_OR_EQUAL,
        }
        if operator in threshold_ops and value is None:
            raise ValueError("threshold_value is required for threshold operators")
        return value


class DataObjectCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    object_id: str | None = None
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
    object_id: str | None = None
    object_name: str
    object_type: ObjectType
    markets: list[StructuredMarket]

    def included_markets(self) -> list[StructuredMarket]:
        return [market for market in self.markets if market.include]


class AggregationResult(BaseModel):
    object_name: str
    object_type: ObjectType
    rows: list[dict[str, Any]]
    chart_json: dict[str, Any]
    anomalies: list[str]


class PointEstimate(BaseModel):
    object_id: str | None = None
    object_name: str
    object_type: ObjectType
    role: Literal["boolean", "time", "numeric", "context"]

    boolean_probability: float | None = None

    median_date: str | None = None

    expected_value: float | None = None
    p10: float | None = None
    p50: float | None = None
    p90: float | None = None
    unit: str | None = None

    top_category: str | None = None
    top_category_probability: float | None = None

    anomalies: list[str] = Field(default_factory=list)
    fit_method: Literal["lognormal", "midpoint", "argmax", "median"]


class OverviewSummary(BaseModel):
    search_run_id: str
    topic: str
    narrative: str
    point_estimates: list[PointEstimate]


class SearchResult(BaseModel):
    search_run_id: str
    topic: str
    search_terms: list[str]
    raw_markets: list[RawMarket]
    extraction: StructuredExtraction


def build_object_id(
    object_name: str,
    object_type: ObjectType,
    event_title: str | None,
    resolution_date: str | None,
    threshold_unit: str | None,
) -> str:
    signature = "|".join(
        [
            object_name.strip().casefold(),
            object_type.value,
            (event_title or "").strip().casefold(),
            (resolution_date or "").split("T")[0],
            (threshold_unit or "").strip().casefold(),
        ]
    )
    slug = re.sub(r"[^a-z0-9]+", "-", object_name.strip().casefold()).strip("-")[:48]
    digest = hashlib.sha1(signature.encode("utf-8")).hexdigest()[:10]
    return f"{slug or 'object'}-{digest}"
