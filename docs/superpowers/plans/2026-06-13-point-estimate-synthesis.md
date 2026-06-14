# Point Estimate Synthesis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce one point estimate per market object plus one AI-generated narrative sentence for the topic, while keeping the full distribution available on demand.

**Architecture:** A new deterministic analysis layer (`parsers.py` + `point_estimate.py`) collapses each object's probability distribution into a single number/date/category via four paths (lognormal EV, median date, argmax, median probability). The AI layer only writes the final narrative from those point estimates. The web layer exposes `POST /summarize/{run_id}` (returns `OverviewSummary` JSON) and `GET /aggregate/{run_id}/{object_id}` (returns `AggregationResult` JSON); a thin `app.js` renders the summary and fetches distribution detail on click.

**Tech Stack:** Python 3.12, FastAPI, Pydantic v2, numpy (least squares), `statistics.NormalDist` (inverse normal CDF — **no scipy**), sqlite, Jinja2, vanilla JS. Tooling: `uv`, `pytest`, `ruff`, `mypy --strict`.

**Conventions (read before starting):**
- Run tests with `uv run pytest` (the repo now sets `pythonpath = ["."]`).
- Immutable style: build new objects, never mutate inputs.
- Every numeric value is stored/fitted in **canonical billions** internally; `PointEstimate.unit` records the display unit and values are converted back for display only.
- After each task: `uv run ruff check app tests` and `uv run mypy app` must pass.

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `app/analysis/parsers.py` | Create | Pure parsing: money-bracket strings, deadline dates, unit canonicalization. No domain logic. |
| `app/analysis/point_estimate.py` | Create | Four-path point-estimate computation + object→(path,role) classifier + AI-input formatter. |
| `app/analysis/aggregators.py` | Modify | Reuse `parsers` for money normalization (DRY); behavior unchanged. |
| `app/models.py` | Modify | Add `PointEstimate`, `OverviewSummary`. |
| `app/ai/prompts.py` | Modify | Add `SYNTHESIZE_OVERVIEW_PROMPT`. |
| `app/ai/provider.py` | Modify | Add abstract `synthesize_overview()`. |
| `app/ai/fallback.py` | Modify | Template-based `synthesize_overview()`. |
| `app/ai/deepseek.py` | Modify | API-based `synthesize_overview()` + `_chat_text()` helper. |
| `app/storage/repositories.py` | Modify | Add `SearchRunRepository.get_topic()`. |
| `app/service.py` | Modify | Add `summarize()` orchestration. |
| `app/web/routes.py` | Modify | Add `GET /summary/{run_id}`, `POST /summarize/{run_id}`, `GET /aggregate/{run_id}/{object_id}`. |
| `app/web/templates/summary.html` | Create | Shell page that hosts `app.js`. |
| `app/web/templates/objects.html` | Modify | Add link to the summary page. |
| `app/web/static/app.js` | Create | Fetch summary + expand distribution detail. |
| `app/web/static/app.css` | Modify | Styles for cards + collapsible detail. |
| `tests/test_parsers.py` | Create | Parser unit tests. |
| `tests/test_point_estimate.py` | Create | All four paths + classifier. |
| `tests/test_ai_provider.py` | Modify | Fallback narrative tests. |
| `tests/test_web_routes.py` | Modify | `/summarize` + `/aggregate` endpoint tests. |

---

## Task 1: Money-bracket and unit parser

**Files:**
- Create: `app/analysis/parsers.py`
- Test: `tests/test_parsers.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_parsers.py`:

```python
from app.analysis.parsers import MoneyBracket, canonical_unit, parse_money_bracket, to_billions


def test_parses_closed_range() -> None:
    assert parse_money_bracket("$1.25–$1.5T") == MoneyBracket(lower=1.25, upper=1.5, unit="T")


def test_parses_open_upper_tail() -> None:
    assert parse_money_bracket("$3.0T+") == MoneyBracket(lower=3.0, upper=None, unit="T")


def test_parses_open_lower_tail() -> None:
    assert parse_money_bracket("<$1.25T") == MoneyBracket(lower=None, upper=1.25, unit="T")


def test_parses_billions_with_hyphen() -> None:
    assert parse_money_bracket("600B+") == MoneyBracket(lower=600.0, upper=None, unit="B")


def test_returns_none_for_non_numeric() -> None:
    assert parse_money_bracket("No IPO") is None
    assert parse_money_bracket("No IPO by December 31, 2027") is None


def test_canonical_unit_maps_verbose_forms() -> None:
    assert canonical_unit("USD billion") == "B"
    assert canonical_unit("USD trillion") == "T"
    assert canonical_unit("USD million") == "M"


def test_to_billions_converts_units() -> None:
    assert to_billions(1.5, "T") == 1500.0
    assert to_billions(600.0, "B") == 600.0
    assert to_billions(500.0, "M") == 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_parsers.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.analysis.parsers'`.

- [ ] **Step 3: Write minimal implementation**

Create `app/analysis/parsers.py`:

```python
import re
from dataclasses import dataclass

UNIT_TO_BILLIONS = {"B": 1.0, "T": 1000.0, "M": 0.001}

_VERBOSE_UNIT = {
    "t": "T",
    "trillion": "T",
    "usd trillion": "T",
    "b": "B",
    "bn": "B",
    "billion": "B",
    "usd billion": "B",
    "m": "M",
    "million": "M",
    "usd million": "M",
}

_NUMBER = r"\$?\s*(\d+(?:\.\d+)?)"
_UNIT = r"\s*(T|B|M|trillion|billion|million)?"
_RANGE_RE = re.compile(_NUMBER + r"\s*[-–—]\s*" + _NUMBER + _UNIT, re.IGNORECASE)
_OPEN_UPPER_RE = re.compile(_NUMBER + r"\s*(T|B|M|trillion|billion|million)?\s*\+", re.IGNORECASE)
_OPEN_LOWER_RE = re.compile(r"[<≤]\s*" + _NUMBER + _UNIT, re.IGNORECASE)
_SINGLE_RE = re.compile(_NUMBER + _UNIT, re.IGNORECASE)


@dataclass(frozen=True)
class MoneyBracket:
    lower: float | None
    upper: float | None
    unit: str


def canonical_unit(raw: str | None) -> str:
    if not raw:
        return "B"
    return _VERBOSE_UNIT.get(raw.strip().casefold(), "B")


def to_billions(value: float, unit: str) -> float:
    return value * UNIT_TO_BILLIONS.get(canonical_unit(unit), 1.0)


def parse_money_bracket(text: str | None) -> MoneyBracket | None:
    if not text:
        return None
    cleaned = text.strip()
    range_match = _RANGE_RE.search(cleaned)
    if range_match:
        unit = canonical_unit(range_match.group(3))
        return MoneyBracket(float(range_match.group(1)), float(range_match.group(2)), unit)
    open_upper = _OPEN_UPPER_RE.search(cleaned)
    if open_upper:
        return MoneyBracket(float(open_upper.group(1)), None, canonical_unit(open_upper.group(2)))
    open_lower = _OPEN_LOWER_RE.search(cleaned)
    if open_lower:
        return MoneyBracket(None, float(open_lower.group(1)), canonical_unit(open_lower.group(2)))
    single = _SINGLE_RE.search(cleaned)
    # Require a currency/unit cue so bare numbers in prose (e.g. the "31, 2027"
    # in "No IPO by December 31, 2027") are NOT mistaken for a money bracket.
    if single and ("$" in cleaned or single.group(2)):
        return MoneyBracket(float(single.group(1)), float(single.group(1)), canonical_unit(single.group(2)))
    return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_parsers.py -q`
Expected: PASS (7 tests). The currency-cue guard in the final branch is what makes `test_returns_none_for_non_numeric` pass — `"No IPO by December 31, 2027"` has digits but no `$`/unit, so `_SINGLE_RE`'s match on `31` is rejected and the function returns `None`. `"600B+"` still parses via `_OPEN_UPPER_RE` before the single branch is reached.

- [ ] **Step 5: Commit**

```bash
git add app/analysis/parsers.py tests/test_parsers.py
git commit -m "feat: add money-bracket and unit parser"
```

---

## Task 2: Deadline-date parser

**Files:**
- Modify: `app/analysis/parsers.py`
- Test: `tests/test_parsers.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_parsers.py`:

```python
from app.analysis.parsers import parse_deadline_date


def test_parses_full_month_name() -> None:
    assert parse_deadline_date("Will Anthropic IPO by October 31, 2026?") == "2026-10-31"


def test_parses_abbreviated_month() -> None:
    assert parse_deadline_date("IPO by Oct 31, 2026") == "2026-10-31"


def test_parses_no_ipo_deadline() -> None:
    assert parse_deadline_date("No IPO by December 31, 2027") == "2027-12-31"


def test_returns_none_without_date() -> None:
    assert parse_deadline_date("Anthropic valuation above $100B") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_parsers.py -k deadline -q`
Expected: FAIL — `ImportError: cannot import name 'parse_deadline_date'`.

- [ ] **Step 3: Write minimal implementation**

Append to `app/analysis/parsers.py`:

```python
_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}
_DATE_RE = re.compile(r"([A-Za-z]{3,9})\s+(\d{1,2}),?\s+(\d{4})")


def parse_deadline_date(text: str | None) -> str | None:
    if not text:
        return None
    match = _DATE_RE.search(text)
    if not match:
        return None
    month = _MONTHS.get(match.group(1)[:3].casefold())
    if month is None:
        return None
    day = int(match.group(2))
    year = int(match.group(3))
    return f"{year:04d}-{month:02d}-{day:02d}"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_parsers.py -q`
Expected: PASS (11 tests).

- [ ] **Step 5: Commit**

```bash
git add app/analysis/parsers.py tests/test_parsers.py
git commit -m "feat: add deadline-date parser"
```

---

## Task 3: PointEstimate and OverviewSummary models

**Files:**
- Modify: `app/models.py` (append after `AggregationResult`, around line 157)
- Test: `tests/test_models.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_models.py`:

```python
from app.models import ObjectType, OverviewSummary, PointEstimate


def test_point_estimate_defaults_optional_fields() -> None:
    estimate = PointEstimate(
        object_name="Anthropic IPO",
        object_type=ObjectType.BOOLEAN,
        role="boolean",
        boolean_probability=0.75,
        anomalies=[],
        fit_method="median",
    )
    assert estimate.boolean_probability == 0.75
    assert estimate.expected_value is None


def test_overview_summary_holds_estimates() -> None:
    summary = OverviewSummary(
        search_run_id="run-1",
        topic="Anthropic IPO",
        narrative="市场预测发生概率为 75%。",
        point_estimates=[],
    )
    assert summary.topic == "Anthropic IPO"
    assert summary.point_estimates == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_models.py -k "point_estimate or overview" -q`
Expected: FAIL — `ImportError: cannot import name 'PointEstimate'`.

- [ ] **Step 3: Write minimal implementation**

In `app/models.py`, add `from typing import Literal` to the typing import line, then append after the `AggregationResult` class:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_models.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/models.py tests/test_models.py
git commit -m "feat: add PointEstimate and OverviewSummary models"
```

---

## Task 4: Numeric path — intervals, lognormal fit, midpoint fallback

This is the core algorithm. It accepts a unit-normalized list of intervals and returns numeric fields. Both categorical money buckets (Task 6) and continuous thresholds (Task 6) feed it.

**Files:**
- Create: `app/analysis/point_estimate.py`
- Test: `tests/test_point_estimate.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_point_estimate.py`:

```python
from app.analysis.point_estimate import Interval, numeric_estimate


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_point_estimate.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.analysis.point_estimate'`.

- [ ] **Step 3: Write minimal implementation**

Create `app/analysis/point_estimate.py`:

```python
import math
from dataclasses import dataclass, field
from statistics import NormalDist

import numpy as np

from app.analysis.parsers import UNIT_TO_BILLIONS, canonical_unit

_NORMAL = NormalDist()
_HIGH_TAIL_THRESHOLD = 0.30
_MIN_R_SQUARED = 0.85


@dataclass(frozen=True)
class Interval:
    lower: float | None  # canonical billions
    upper: float | None  # canonical billions
    probability: float


@dataclass
class NumericEstimate:
    expected_value: float | None
    p10: float | None
    p50: float | None
    p90: float | None
    unit: str
    fit_method: str
    anomalies: list[str] = field(default_factory=list)


def numeric_estimate(intervals: list[Interval], unit: str) -> NumericEstimate:
    total = sum(item.probability for item in intervals)
    normalized = [
        Interval(item.lower, item.upper, item.probability / total)
        for item in intervals
        if total > 0
    ]
    anomalies: list[str] = []
    if len(normalized) < 2:
        anomalies.append("insufficient_data")
        return NumericEstimate(None, None, None, None, unit, "midpoint", anomalies)

    tail = next((item for item in normalized if item.upper is None), None)
    if tail is not None and tail.probability > _HIGH_TAIL_THRESHOLD:
        anomalies.append("high_tail_probability")

    fit = _fit_lognormal(normalized)
    if fit is not None:
        mu, sigma = fit
        return NumericEstimate(
            expected_value=_from_billions(math.exp(mu + sigma * sigma / 2), unit),
            p10=_from_billions(math.exp(mu + sigma * _NORMAL.inv_cdf(0.10)), unit),
            p50=_from_billions(math.exp(mu + sigma * _NORMAL.inv_cdf(0.50)), unit),
            p90=_from_billions(math.exp(mu + sigma * _NORMAL.inv_cdf(0.90)), unit),
            unit=unit,
            fit_method="lognormal",
            anomalies=anomalies,
        )

    anomalies.append("lognormal_fit_failed")
    expected = _midpoint_expected_value(normalized)
    return NumericEstimate(
        expected_value=_from_billions(expected, unit),
        p10=None,
        p50=_from_billions(expected, unit),
        p90=None,
        unit=unit,
        fit_method="midpoint",
        anomalies=anomalies,
    )


def _fit_lognormal(intervals: list[Interval]) -> tuple[float, float] | None:
    cumulative = 0.0
    thresholds: list[float] = []
    cdfs: list[float] = []
    for item in sorted(intervals, key=lambda i: (i.upper is None, i.upper or 0.0)):
        cumulative += item.probability
        if item.upper is not None and item.upper > 0 and 0.0 < cumulative < 1.0:
            thresholds.append(item.upper)
            cdfs.append(cumulative)
    if len(thresholds) < 3:
        return None
    y = np.log(np.array(thresholds))
    z = np.array([_NORMAL.inv_cdf(c) for c in cdfs])
    # Index rather than unpack the ndarray so mypy --strict doesn't demand an
    # annotation for the iterable. polyfit(deg=1) returns [slope, intercept].
    coefficients = np.polyfit(z, y, 1)
    sigma = float(coefficients[0])
    mu = float(coefficients[1])
    if sigma <= 0:
        return None
    predicted = mu + sigma * z
    ss_res = float(np.sum((y - predicted) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    if r_squared < _MIN_R_SQUARED:
        return None
    return mu, sigma


def _midpoint_expected_value(intervals: list[Interval]) -> float:
    total = 0.0
    for item in intervals:
        if item.lower is not None and item.upper is not None:
            representative = (item.lower + item.upper) / 2
        elif item.upper is not None:  # open lower tail
            representative = item.upper / 2
        elif item.lower is not None:  # open upper tail
            representative = item.lower * 1.5
        else:
            representative = 0.0
        total += item.probability * representative
    return total


def _from_billions(value: float, unit: str) -> float:
    factor = UNIT_TO_BILLIONS.get(canonical_unit(unit), 1.0)
    return round(value / factor, 4)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_point_estimate.py -q`
Expected: PASS (4 tests). If the lognormal test's R² dips below 0.85 it falls back to midpoint — the chosen probabilities above are symmetric in log space and should fit; if it fails, widen the assertion to accept `fit_method in {"lognormal", "midpoint"}` only after confirming the math, not before.

- [ ] **Step 5: Commit**

```bash
git add app/analysis/point_estimate.py tests/test_point_estimate.py
git commit -m "feat: add numeric point estimate with lognormal fit and midpoint fallback"
```

---

## Task 5: Boolean, argmax, and date paths

**Files:**
- Modify: `app/analysis/point_estimate.py`
- Test: `tests/test_point_estimate.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_point_estimate.py`:

```python
import pytest

from app.analysis.point_estimate import (
    argmax_estimate,
    boolean_estimate,
    median_date_estimate,
)


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
    date, anomalies = median_date_estimate([("2026-06-30", 0.2), ("2026-10-31", 0.4), ("2027-03-31", 0.3)])
    assert date == "2026-10-31"
    assert anomalies == []


def test_median_date_not_reached() -> None:
    date, anomalies = median_date_estimate([("2026-06-30", 0.2), ("2026-10-31", 0.2)])
    assert date == "2026-10-31"
    assert "median_date_not_reached" in anomalies
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_point_estimate.py -k "boolean or argmax or median_date" -q`
Expected: FAIL — `ImportError: cannot import name 'boolean_estimate'`.

- [ ] **Step 3: Write minimal implementation**

Append to `app/analysis/point_estimate.py` (add `from statistics import median, pstdev` to the existing `statistics` import line — i.e. `from statistics import NormalDist, median, pstdev`):

```python
_HETEROGENEOUS_STD = 0.30


def boolean_estimate(probabilities: list[float]) -> tuple[float | None, list[str]]:
    if not probabilities:
        return None, ["insufficient_data"]
    anomalies: list[str] = []
    if len(probabilities) > 1 and pstdev(probabilities) > _HETEROGENEOUS_STD:
        anomalies.append("heterogeneous_group")
    return round(median(probabilities), 6), anomalies


def argmax_estimate(grouped: dict[str, list[float]]) -> tuple[str | None, float | None]:
    if not grouped:
        return None, None
    averages = {name: sum(values) / len(values) for name, values in grouped.items()}
    total = sum(averages.values())
    if total <= 0:
        return None, None
    normalized = {name: value / total for name, value in averages.items()}
    top = max(normalized, key=lambda name: normalized[name])
    return top, round(normalized[top], 6)


def median_date_estimate(
    dated_probabilities: list[tuple[str, float]],
) -> tuple[str | None, list[str]]:
    if not dated_probabilities:
        return None, ["insufficient_data"]
    ordered = sorted(dated_probabilities, key=lambda item: item[0])
    cumulative = 0.0
    for date, probability in ordered:
        cumulative += probability
        if cumulative >= 0.5:
            return date, []
    return ordered[-1][0], ["median_date_not_reached"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_point_estimate.py -q`
Expected: PASS (9 tests total).

- [ ] **Step 5: Commit**

```bash
git add app/analysis/point_estimate.py tests/test_point_estimate.py
git commit -m "feat: add boolean, argmax, and median-date point estimates"
```

---

## Task 6: Classifier and `compute_point_estimate` dispatcher

Routes a list of `StructuredMarket` for one object into the right path and returns a `PointEstimate`. Continuous-threshold objects are routed into the numeric path (Path 1) by deriving intervals from `threshold_value` + `operator`.

**Files:**
- Modify: `app/analysis/point_estimate.py`
- Test: `tests/test_point_estimate.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_point_estimate.py`:

```python
from app.analysis.point_estimate import compute_point_estimate
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
    markets = [_market(object_type=ObjectType.BOOLEAN, operator=Operator.EQUAL, category_value="yes", probability=0.75)]
    estimate = compute_point_estimate("IPO", ObjectType.BOOLEAN, markets)
    assert estimate.role == "boolean"
    assert estimate.boolean_probability == 0.75


def test_time_uses_resolution_date() -> None:
    markets = [
        _market(market_id="a", object_type=ObjectType.TIME, category_value="by Q2", probability=0.3, resolution_date="2026-06-30"),
        _market(market_id="b", object_type=ObjectType.TIME, category_value="by Q4", probability=0.5, resolution_date="2026-12-31"),
    ]
    estimate = compute_point_estimate("Timing", ObjectType.TIME, markets)
    assert estimate.role == "time"
    assert estimate.median_date == "2026-12-31"


def test_continuous_thresholds_route_to_numeric() -> None:
    markets = [
        _market(market_id="a", object_type=ObjectType.CONTINUOUS, operator=Operator.GREATER_THAN, threshold_value=85.0, threshold_unit="USD billion", probability=0.70),
        _market(market_id="b", object_type=ObjectType.CONTINUOUS, operator=Operator.GREATER_THAN, threshold_value=100.0, threshold_unit="USD billion", probability=0.45),
        _market(market_id="c", object_type=ObjectType.CONTINUOUS, operator=Operator.GREATER_THAN, threshold_value=150.0, threshold_unit="USD billion", probability=0.10),
    ]
    estimate = compute_point_estimate("Valuation", ObjectType.CONTINUOUS, markets)
    assert estimate.role == "numeric"
    assert estimate.unit == "B"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_point_estimate.py -k compute -q`
Expected: FAIL — `ImportError: cannot import name 'compute_point_estimate'`.

- [ ] **Step 3: Write minimal implementation**

Append to `app/analysis/point_estimate.py` (add imports at top: `from collections import defaultdict` and `from app.models import ObjectType, Operator, PointEstimate, StructuredMarket`). Note: the aggregator already exposes `threshold_cdf_probability`; import it to avoid duplicating the operator-to-CDF rule:

```python
from app.analysis.aggregators import threshold_cdf_probability
from app.analysis.parsers import (
    MoneyBracket,
    parse_deadline_date,
    parse_money_bracket,
    to_billions,
)

_RESIDUAL_MASS_THRESHOLD = 0.05


def compute_point_estimate(
    object_name: str,
    object_type: ObjectType,
    markets: list[StructuredMarket],
) -> PointEstimate:
    included = [market for market in markets if market.include]
    object_id = included[0].object_id if included else None
    if object_type == ObjectType.BOOLEAN:
        estimate = _boolean_point_estimate(object_name, object_type, included)
    elif object_type == ObjectType.TIME:
        estimate = _date_point_estimate(object_name, object_type, included)
    elif object_type == ObjectType.CONTINUOUS:
        estimate = _continuous_point_estimate(object_name, object_type, included)
    else:
        estimate = _categorical_point_estimate(object_name, object_type, included)
    # Stamp the source object_id so the frontend can fetch the matching
    # distribution unambiguously even when two objects share a display name.
    return estimate.model_copy(update={"object_id": object_id})


def _boolean_point_estimate(
    object_name: str, object_type: ObjectType, markets: list[StructuredMarket]
) -> PointEstimate:
    value, anomalies = boolean_estimate([m.probability for m in markets])
    return PointEstimate(
        object_name=object_name,
        object_type=object_type,
        role="boolean",
        boolean_probability=value,
        anomalies=anomalies,
        fit_method="median",
    )


def _date_point_estimate(
    object_name: str, object_type: ObjectType, markets: list[StructuredMarket]
) -> PointEstimate:
    dated: list[tuple[str, float]] = []
    for market in markets:
        date = (market.resolution_date or "").split("T")[0] or parse_deadline_date(
            market.category_value or market.question
        )
        if date:
            dated.append((date, market.probability))
    median_date, anomalies = median_date_estimate(dated)
    return PointEstimate(
        object_name=object_name,
        object_type=object_type,
        role="time",
        median_date=median_date,
        anomalies=anomalies,
        fit_method="median",
    )


def _categorical_point_estimate(
    object_name: str, object_type: ObjectType, markets: list[StructuredMarket]
) -> PointEstimate:
    brackets = [(parse_money_bracket(m.category_value), m.probability) for m in markets]
    parseable = [(bracket, prob) for bracket, prob in brackets if bracket is not None]
    if len(parseable) >= 2:
        dropped_mass = sum(prob for bracket, prob in brackets if bracket is None)
        return _numeric_from_brackets(object_name, object_type, parseable, dropped_mass)

    grouped: dict[str, list[float]] = defaultdict(list)
    for market in markets:
        grouped[market.category_value or market.question].append(market.probability)
    top, probability = argmax_estimate(grouped)
    return PointEstimate(
        object_name=object_name,
        object_type=object_type,
        role="context",
        top_category=top,
        top_category_probability=probability,
        anomalies=[],
        fit_method="argmax",
    )


def _numeric_from_brackets(
    object_name: str,
    object_type: ObjectType,
    parseable: list[tuple[MoneyBracket, float]],
    dropped_mass: float = 0.0,
) -> PointEstimate:
    unit = _dominant_unit([bracket.unit for bracket, _ in parseable])
    intervals = [
        Interval(
            lower=to_billions(bracket.lower, bracket.unit) if bracket.lower is not None else None,
            upper=to_billions(bracket.upper, bracket.unit) if bracket.upper is not None else None,
            probability=prob,
        )
        for bracket, prob in parseable
    ]
    estimate = numeric_estimate(intervals, unit)
    point = _numeric_to_point_estimate(object_name, object_type, estimate)
    # Non-numeric buckets (e.g. "No IPO") are excluded from the fit; if their
    # combined probability is material, flag it rather than silently dropping it.
    if dropped_mass > _RESIDUAL_MASS_THRESHOLD:
        return point.model_copy(update={"anomalies": [*point.anomalies, "residual_mass_dropped"]})
    return point


def _continuous_point_estimate(
    object_name: str, object_type: ObjectType, markets: list[StructuredMarket]
) -> PointEstimate:
    points = sorted(
        [
            (
                to_billions(m.threshold_value, m.threshold_unit or "B"),
                threshold_cdf_probability(m.operator, m.probability),
            )
            for m in markets
            if m.threshold_value is not None
        ],
        key=lambda item: item[0],
    )
    intervals: list[Interval] = []
    previous_threshold: float | None = None
    previous_cdf = 0.0
    for threshold, cdf in points:
        intervals.append(
            Interval(lower=previous_threshold, upper=threshold, probability=max(cdf - previous_cdf, 0.0))
        )
        previous_threshold = threshold
        previous_cdf = cdf
    if points:
        intervals.append(
            Interval(lower=points[-1][0], upper=None, probability=max(1.0 - points[-1][1], 0.0))
        )
    estimate = numeric_estimate(intervals, "B")
    return _numeric_to_point_estimate(object_name, object_type, estimate)


def _numeric_to_point_estimate(
    object_name: str, object_type: ObjectType, estimate: NumericEstimate
) -> PointEstimate:
    return PointEstimate(
        object_name=object_name,
        object_type=object_type,
        role="numeric",
        expected_value=estimate.expected_value,
        p10=estimate.p10,
        p50=estimate.p50,
        p90=estimate.p90,
        unit=estimate.unit,
        anomalies=estimate.anomalies,
        fit_method=estimate.fit_method,  # type: ignore[arg-type]
    )


def _dominant_unit(units: list[str]) -> str:
    if not units:
        return "B"
    return max(set(units), key=units.count)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_point_estimate.py -q`
Expected: PASS (14 tests total). Then `uv run mypy app` — fix any `type: ignore` placement the checker rejects.

- [ ] **Step 5: Commit**

```bash
git add app/analysis/point_estimate.py tests/test_point_estimate.py
git commit -m "feat: add point estimate classifier and dispatcher"
```

---

## Task 7: DRY refactor — aggregators reuse the shared parser

`aggregators.py` has its own `normalize_money_value`. Point it at `parsers.to_billions` so there is one money model. Keep every existing aggregator test green.

**Files:**
- Modify: `app/analysis/aggregators.py:177-182`
- Test: `tests/test_aggregators.py` (existing — must stay green)

- [ ] **Step 1: Confirm the existing tests pass before touching anything**

Run: `uv run pytest tests/test_aggregators.py -q`
Expected: PASS (4 tests).

- [ ] **Step 2: Replace the local normalizer with the shared one**

In `app/analysis/aggregators.py`, add to the imports:

```python
from app.analysis.parsers import to_billions
```

Replace `normalize_money_value` (lines ~177-182) and its call site in `category_sort_key`:

```python
def normalize_money_value(number_text: str, unit_text: str | None) -> float:
    return to_billions(float(number_text), unit_text or "B")
```

(The regex in `category_sort_key` yields units like `t`/`trillion`/`b`/`billion`; `to_billions` already maps those via `canonical_unit`, so no other change is needed.)

- [ ] **Step 3: Run the full analysis test suite**

Run: `uv run pytest tests/test_aggregators.py tests/test_point_estimate.py tests/test_parsers.py -q`
Expected: PASS. The key regression guard is `test_categorical_money_buckets_sort_and_normalize` (ordering of `<$1.25T`, `$1.25–$1.5T`, `$2.0–$2.25T`, `$3.0T+`, `No IPO...`).

- [ ] **Step 4: Lint and type-check**

Run: `uv run ruff check app && uv run mypy app`
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add app/analysis/aggregators.py
git commit -m "refactor: aggregators reuse shared money parser"
```

---

## Task 8: AI narrative — prompt, protocol, fallback, DeepSeek

**Files:**
- Modify: `app/ai/prompts.py`, `app/ai/provider.py`, `app/ai/fallback.py`, `app/ai/deepseek.py`
- Modify: `app/analysis/point_estimate.py` (add `describe_point_estimates` formatter)
- Test: `tests/test_ai_provider.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_ai_provider.py`:

```python
from app.ai.fallback import RuleBasedProvider
from app.models import ObjectType, PointEstimate


async def test_fallback_narrative_boolean_time_numeric() -> None:
    provider = RuleBasedProvider()
    estimates = [
        PointEstimate(object_name="IPO", object_type=ObjectType.BOOLEAN, role="boolean", boolean_probability=0.75, anomalies=[], fit_method="median"),
        PointEstimate(object_name="Timing", object_type=ObjectType.TIME, role="time", median_date="2026-10-31", anomalies=[], fit_method="median"),
        PointEstimate(object_name="估值", object_type=ObjectType.CATEGORICAL, role="numeric", expected_value=1.6, unit="T", p50=1.6, anomalies=[], fit_method="lognormal"),
    ]

    narrative = await provider.synthesize_overview("Anthropic IPO", estimates)

    assert "75%" in narrative
    assert "2026-10-31" in narrative
    assert "1.6" in narrative


async def test_fallback_narrative_categorical_only() -> None:
    provider = RuleBasedProvider()
    estimates = [
        PointEstimate(object_name="Top lab", object_type=ObjectType.CATEGORICAL, role="context", top_category="Anthropic", top_category_probability=0.62, anomalies=[], fit_method="argmax"),
    ]

    narrative = await provider.synthesize_overview("Top lab", estimates)

    assert "Anthropic" in narrative
    assert "62%" in narrative
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_ai_provider.py -k narrative -q`
Expected: FAIL — `AttributeError: 'RuleBasedProvider' object has no attribute 'synthesize_overview'`.

- [ ] **Step 3: Write minimal implementation**

(a) In `app/ai/prompts.py`, append:

```python
SYNTHESIZE_OVERVIEW_PROMPT = """You are a prediction market analyst. Distill the market data into one or two sentences. Use conditional language when appropriate ("if X happens…"). Flag high uncertainty when anomalies are present. Give the conclusion directly — do not explain the method. Answer in the same language as the topic.
"""
```

(b) In `app/ai/provider.py`, add the import and abstract method:

```python
from app.models import PointEstimate, RawMarket, StructuredExtractionDraft
```

```python
    @abstractmethod
    async def synthesize_overview(
        self, topic: str, point_estimates: list[PointEstimate]
    ) -> str:
        raise NotImplementedError
```

(c) In `app/analysis/point_estimate.py`, append a deterministic formatter used by both providers:

```python
def describe_point_estimates(point_estimates: list[PointEstimate]) -> str:
    lines: list[str] = []
    for estimate in _ordered_by_role(point_estimates):
        if estimate.role == "boolean" and estimate.boolean_probability is not None:
            line = f"- [boolean] {estimate.object_name}: {round(estimate.boolean_probability * 100)}%"
        elif estimate.role == "time" and estimate.median_date:
            line = f"- [time] {estimate.object_name}: median {estimate.median_date}"
        elif estimate.role == "numeric" and estimate.expected_value is not None:
            line = f"- [numeric] {estimate.object_name}: ~{estimate.expected_value}{estimate.unit or ''}"
        elif estimate.role == "context" and estimate.top_category:
            line = f"- [context] {estimate.object_name}: {estimate.top_category} ({round((estimate.top_category_probability or 0) * 100)}%)"
        else:
            # No representable value (e.g. a TIME object with no resolvable date).
            # Still emit a line so anomalies attach to THIS estimate, never to the
            # previous one — and never index an empty `lines` list.
            line = f"- [{estimate.role}] {estimate.object_name}: insufficient data"
        if estimate.anomalies:
            line += f" [anomalies: {', '.join(estimate.anomalies)}]"
        lines.append(line)
    return "\n".join(lines)


_ROLE_ORDER = {"boolean": 0, "time": 1, "numeric": 2, "context": 3}


def _ordered_by_role(point_estimates: list[PointEstimate]) -> list[PointEstimate]:
    return sorted(point_estimates, key=lambda pe: _ROLE_ORDER.get(pe.role, 9))
```

(d) In `app/ai/fallback.py`, add imports and the method on `RuleBasedProvider`:

```python
from app.analysis.point_estimate import _ordered_by_role
from app.models import PointEstimate
```

```python
    async def synthesize_overview(
        self, topic: str, point_estimates: list[PointEstimate]
    ) -> str:
        ordered = _ordered_by_role(point_estimates)
        by_role = {estimate.role: estimate for estimate in reversed(ordered)}
        boolean = by_role.get("boolean")
        time = by_role.get("time")
        numeric = by_role.get("numeric")
        context = by_role.get("context")

        if boolean and boolean.boolean_probability is not None:
            pct = round(boolean.boolean_probability * 100)
            if time and time.median_date and numeric and numeric.expected_value is not None:
                value = f"{numeric.expected_value}{numeric.unit or ''}"
                return f"市场预测{topic}发生概率为{pct}%，最可能在{time.median_date}；若发生，预期{numeric.object_name}约{value}。"
            if time and time.median_date:
                return f"市场预测{topic}发生概率为{pct}%，最可能在{time.median_date}。"
            return f"市场预测{topic}发生概率为{pct}%。"
        if context and context.top_category:
            pct = round((context.top_category_probability or 0) * 100)
            return f"市场认为{topic}最可能的结果是{context.top_category}（{pct}%概率）。"
        if numeric and numeric.expected_value is not None:
            return f"市场预期{numeric.object_name}约{numeric.expected_value}{numeric.unit or ''}。"
        return f"暂无足够的市场数据来总结{topic}。"
```

(e) In `app/ai/deepseek.py`, add imports and methods:

```python
from app.analysis.point_estimate import describe_point_estimates
from app.ai.prompts import SEARCH_PLANNING_PROMPT, STRUCTURE_MARKETS_PROMPT, SYNTHESIZE_OVERVIEW_PROMPT
from app.models import PointEstimate, RawMarket, StructuredExtractionDraft
```

```python
    async def synthesize_overview(
        self, topic: str, point_estimates: list[PointEstimate]
    ) -> str:
        user_content = (
            f"Topic: {topic}\n"
            f"Point estimates (ordered by importance):\n{describe_point_estimates(point_estimates)}\n\n"
            "Summarize the market's view in one or two sentences."
        )
        return await self._chat_text(SYNTHESIZE_OVERVIEW_PROMPT, user_content)

    async def _chat_text(self, system_prompt: str, user_content: str) -> str:
        response = await self.http_client.post(
            "https://api.deepseek.com/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
            },
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        return str(content).strip()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_ai_provider.py -q && uv run mypy app`
Expected: PASS + clean types. (Importing `_ordered_by_role` across modules is intentional reuse; if mypy/ruff objects to the leading underscore, rename it to `order_by_role` in `point_estimate.py` and update both call sites.)

- [ ] **Step 5: Commit**

```bash
git add app/ai/prompts.py app/ai/provider.py app/ai/fallback.py app/ai/deepseek.py app/analysis/point_estimate.py tests/test_ai_provider.py
git commit -m "feat: add AI narrative synthesis for point estimates"
```

---

## Task 9: `SearchRunRepository.get_topic` and `OverviewService.summarize`

**Files:**
- Modify: `app/storage/repositories.py:13-24`
- Modify: `app/service.py`
- Test: `tests/test_web_routes.py` (integration test added in Task 11)

- [ ] **Step 1: Write the failing test**

Create a focused service test at `tests/test_summarize.py`:

```python
import pytest

from app.config import get_settings
from app.models import ObjectType, Operator, StructuredMarket
from app.service import OverviewService
from app.storage.db import connect, migrate
from app.storage.repositories import SearchRunRepository, StructuredMarketRepository


def _market(object_id: str, category_value: str, probability: float) -> StructuredMarket:
    return StructuredMarket(
        object_id=object_id, market_id=category_value, question="Q?", event_title="E",
        object_name="Valuation", object_type=ObjectType.CATEGORICAL, operator=Operator.CATEGORY,
        threshold_value=None, threshold_unit=None, category_value=category_value,
        probability=probability, probability_source="outcome_price", resolution_date=None,
        include=True, confidence=0.9, explanation="t",
    )


async def test_summarize_returns_narrative_and_estimates(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("POLYMARKET_OVERVIEW_DB_PATH", str(tmp_path / "s.sqlite"))
    get_settings.cache_clear()
    db = connect(get_settings().db_path)
    migrate(db)
    run_id = SearchRunRepository(db).create("Anthropic valuation", "fallback", ["x"])
    StructuredMarketRepository(db).save_many(run_id, [
        _market("v", "<$1.25T", 0.15),
        _market("v", "$1.25–$1.5T", 0.35),
        _market("v", "$1.5–$2.25T", 0.35),
        _market("v", "$3.0T+", 0.15),
    ])

    service = OverviewService(get_settings())
    summary = await service.summarize(run_id)

    assert summary.topic == "Anthropic valuation"
    assert summary.narrative
    assert summary.point_estimates[0].role == "numeric"
    get_settings.cache_clear()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_summarize.py -q`
Expected: FAIL — `AttributeError: 'OverviewService' object has no attribute 'summarize'`.

- [ ] **Step 3: Write minimal implementation**

(a) In `app/storage/repositories.py`, add to `SearchRunRepository`:

```python
    def get_topic(self, search_run_id: str) -> str | None:
        row = self.db.execute(
            "select topic from search_runs where id = ?",
            (search_run_id,),
        ).fetchone()
        return str(row[0]) if row else None
```

(b) In `app/service.py`, add imports and the method:

```python
from app.analysis.point_estimate import compute_point_estimate
from app.models import (
    AggregationInput,
    AggregationResult,
    ObjectType,
    OverviewSummary,
    PointEstimate,
    SearchResult,
    StructuredMarket,
)
from app.storage.repositories import (
    AggregationRunRepository,
    RawMarketRepository,
    SearchRunRepository,
    StructuredMarketRepository,
)
```

```python
    async def summarize(self, search_run_id: str) -> OverviewSummary:
        db = connect(self.settings.db_path)
        migrate(db)
        topic = SearchRunRepository(db).get_topic(search_run_id)
        if topic is None:
            raise ValueError(f"Unknown search run {search_run_id}")
        markets = [
            market
            for market in StructuredMarketRepository(db).list_effective_for_run(search_run_id)
            if market.include
        ]
        estimates: list[PointEstimate] = []
        for object_id in _ordered_object_ids(markets):
            members = [m for m in markets if (m.object_id or m.object_name) == object_id]
            object_type = members[0].object_type
            if any(m.object_type != object_type for m in members):
                object_type = ObjectType.CATEGORICAL
            estimates.append(compute_point_estimate(members[0].object_name, object_type, members))
        narrative = await self.ai_provider.synthesize_overview(topic, estimates)
        return OverviewSummary(
            search_run_id=search_run_id,
            topic=topic,
            narrative=narrative,
            point_estimates=estimates,
        )
```

Add module-level helper at the bottom of `app/service.py`:

```python
def _ordered_object_ids(markets: list[StructuredMarket]) -> list[str]:
    seen: dict[str, None] = {}
    for market in markets:
        seen.setdefault(market.object_id or market.object_name, None)
    return list(seen)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_summarize.py -q && uv run mypy app`
Expected: PASS + clean.

- [ ] **Step 5: Commit**

```bash
git add app/storage/repositories.py app/service.py tests/test_summarize.py
git commit -m "feat: add summarize orchestration and topic lookup"
```

---

## Task 10: Web routes — `/summarize` and `/aggregate` JSON, `/summary` page

**Files:**
- Modify: `app/web/routes.py`
- Create: `app/web/templates/summary.html`
- Test: `tests/test_web_routes.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_web_routes.py`:

```python
import pytest

from app.analysis.point_estimate import compute_point_estimate  # noqa: F401  (ensures import wiring)


def _seed_run(tmp_path, monkeypatch):
    monkeypatch.setenv("POLYMARKET_OVERVIEW_DB_PATH", str(tmp_path / "w.sqlite"))
    get_settings.cache_clear()
    db = connect(get_settings().db_path)
    migrate(db)
    run_id = SearchRunRepository(db).create("Anthropic valuation", "fallback", ["x"])
    StructuredMarketRepository(db).save_many(run_id, [
        StructuredMarket(object_id="v", market_id="a", question="Q?", event_title="E",
            object_name="Valuation", object_type=ObjectType.CATEGORICAL, operator=Operator.CATEGORY,
            threshold_value=None, threshold_unit=None, category_value="$1.25–$1.5T", probability=0.5,
            probability_source="outcome_price", resolution_date=None, include=True, confidence=0.9, explanation="t"),
        StructuredMarket(object_id="v", market_id="b", question="Q?", event_title="E",
            object_name="Valuation", object_type=ObjectType.CATEGORICAL, operator=Operator.CATEGORY,
            threshold_value=None, threshold_unit=None, category_value="$1.5–$2.25T", probability=0.5,
            probability_source="outcome_price", resolution_date=None, include=True, confidence=0.9, explanation="t"),
    ])
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_web_routes.py -k "summarize_endpoint or aggregate_endpoint" -q`
Expected: FAIL — 404 (routes not defined yet).

- [ ] **Step 3: Write minimal implementation**

In `app/web/routes.py`, add the import and three handlers:

```python
from app.models import OverviewSummary  # add to existing models import line
```

```python
@router.get("/summary/{search_run_id}")
async def summary_page(request: Request, search_run_id: str) -> Response:
    return templates.TemplateResponse(
        request,
        "summary.html",
        {"title": "Overview", "search_run_id": search_run_id},
    )


@router.post("/summarize/{search_run_id}")
async def summarize_run(search_run_id: str) -> OverviewSummary:
    service = OverviewService(get_settings())
    return await service.summarize(search_run_id)


@router.get("/aggregate/{search_run_id}/{object_id}")
async def aggregate_detail(search_run_id: str, object_id: str) -> AggregationResult:
    service = OverviewService(get_settings())
    return service.aggregate_object(search_run_id, object_id)
```

Add to the `app.models` import line: `AggregationResult`. Returning a Pydantic model from a FastAPI handler serializes it to JSON automatically.

> **Known cost (accepted for MVP):** the frontend `init()` issues `POST /summarize/{run_id}` on every page load, and `summarize()` calls `ai_provider.synthesize_overview` each time — so on the DeepSeek path each refresh spends one API call with no caching or persistence. Acceptable for the MVP; a follow-up should persist the `OverviewSummary` (e.g. an `overview_summaries` table keyed by `search_run_id`) and serve a cached copy on subsequent loads.

Create `app/web/templates/summary.html`:

```html
{% extends "base.html" %}
{% block content %}
<section class="panel">
  <h1>Market Overview</h1>
  <p class="muted">run {{ search_run_id }}</p>
  <div id="narrative" class="narrative">Loading…</div>
  <div id="estimates"></div>
</section>
<script>window.SEARCH_RUN_ID = "{{ search_run_id }}";</script>
<script src="/static/app.js"></script>
{% endblock %}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_web_routes.py -q && uv run mypy app`
Expected: PASS + clean.

- [ ] **Step 5: Commit**

```bash
git add app/web/routes.py app/web/templates/summary.html tests/test_web_routes.py
git commit -m "feat: add summarize and aggregate JSON routes"
```

---

## Task 11: Frontend — fetch summary, expand distribution on click

**Files:**
- Create: `app/web/static/app.js`
- Modify: `app/web/templates/objects.html:6` (add a link to the summary page)
- Modify: `app/web/static/app.css` (append styles)

- [ ] **Step 1: Add the entry link**

In `app/web/templates/objects.html`, after the existing review link (line 6), add:

```html
  <p><a class="button-link" href="/summary/{{ run.search_run_id }}">Generate overview</a></p>
```

- [ ] **Step 2: Write `app.js`**

Create `app/web/static/app.js`:

```javascript
const runId = window.SEARCH_RUN_ID;

function fmtEstimate(estimate) {
  if (estimate.role === "boolean") return `${Math.round((estimate.boolean_probability ?? 0) * 100)}%`;
  if (estimate.role === "time") return estimate.median_date ?? "—";
  if (estimate.role === "numeric") return estimate.expected_value != null ? `~${estimate.expected_value}${estimate.unit ?? ""}` : "—";
  if (estimate.role === "context") return `${estimate.top_category} (${Math.round((estimate.top_category_probability ?? 0) * 100)}%)`;
  return "—";
}

async function renderDetail(container, objectId) {
  container.textContent = "Loading…";
  const res = await fetch(`/aggregate/${runId}/${encodeURIComponent(objectId)}`);
  if (!res.ok) { container.textContent = "Failed to load detail."; return; }
  const data = await res.json();
  const head = Object.keys(data.rows[0] ?? {});
  const rows = data.rows.map((row) => `<tr>${head.map((k) => `<td>${row[k]}</td>`).join("")}</tr>`).join("");
  container.innerHTML = `<table><thead><tr>${head.map((k) => `<th>${k}</th>`).join("")}</tr></thead><tbody>${rows}</tbody></table>`;
}

function estimateCard(estimate) {
  const card = document.createElement("div");
  card.className = "estimate-card";
  const objectId = estimate.object_id ?? estimate.object_name;
  card.innerHTML = `<div class="estimate-head"><strong>${estimate.object_name}</strong> <span>${fmtEstimate(estimate)}</span>` +
    (estimate.anomalies.length ? ` <span class="warning">${estimate.anomalies.join(", ")}</span>` : "") +
    `</div><button class="expand">Show distribution</button><div class="detail"></div>`;
  const detail = card.querySelector(".detail");
  card.querySelector(".expand").addEventListener("click", () => {
    detail.classList.toggle("open");
    if (detail.classList.contains("open") && !detail.dataset.loaded) {
      detail.dataset.loaded = "1";
      renderDetail(detail, objectId);
    }
  });
  return card;
}

async function init() {
  const res = await fetch(`/summarize/${runId}`, { method: "POST" });
  const summary = await res.json();
  document.getElementById("narrative").textContent = summary.narrative;
  const container = document.getElementById("estimates");
  summary.point_estimates.forEach((estimate) => container.appendChild(estimateCard(estimate)));
}

init();
```

Note: the detail fetch prefers `estimate.object_id` (stamped by `compute_point_estimate`) and falls back to `object_name`. `aggregate_object` matches either (see `app/service.py:68`), so using the id removes the ambiguity that arises when two objects share a display name.

- [ ] **Step 3: Append styles**

Append to `app/web/static/app.css`:

```css
.narrative { font-size: 1.1rem; margin: 1rem 0; padding: 1rem; background: #f5f7fa; border-radius: 8px; }
.estimate-card { border: 1px solid #e2e6ea; border-radius: 8px; padding: 0.75rem 1rem; margin-bottom: 0.75rem; }
.estimate-head { display: flex; gap: 0.75rem; align-items: center; }
.estimate-card .expand { margin-top: 0.5rem; }
.estimate-card .detail { display: none; margin-top: 0.5rem; }
.estimate-card .detail.open { display: block; }
```

- [ ] **Step 4: Manual smoke test**

Run: `uv run uvicorn app.main:app --port 8011` (background), then in a browser open `http://localhost:8011/summary/<an-existing-run-id>` (use a run id from a prior `/search`, or seed one). Confirm: narrative renders, each estimate shows a value, "Show distribution" expands a table fetched from `/aggregate`. Stop the server.

Expected: no console errors; one network call to `/summarize` on load, one to `/aggregate/...` per first expand.

- [ ] **Step 5: Commit**

```bash
git add app/web/static/app.js app/web/static/app.css app/web/templates/objects.html
git commit -m "feat: add overview summary page with on-demand distribution detail"
```

---

## Task 12: Full-suite verification

**Files:** none (verification only)

- [ ] **Step 1: Run the full test suite**

Run: `uv run pytest -q`
Expected: PASS (original 19 + all new tests; no failures, no collection errors).

- [ ] **Step 2: Lint and type-check the whole project**

Run: `uv run ruff check app tests && uv run mypy app`
Expected: `All checks passed!` and `Success: no issues found`.

- [ ] **Step 3: Confirm no dead `chart_json` regression decision**

Decide per Open Question #7: the detail view renders `rows` (not `chart_json`). `chart_json` remains computed but unused by the new JS. Either leave it (documented) or open a follow-up to remove it. No code change required here — just confirm the JS reads `data.rows`, not `data.chart_json`.

- [ ] **Step 4: Commit any final fixes**

```bash
git add -A
git commit -m "test: verify full point-estimate synthesis suite green"
```

---

## Self-Review

**1. Spec coverage** (against `2026-06-13-point-estimate-synthesis-design.md`):
- Numeric bracket parser → Task 1. Date parser → Task 2. ✓
- Path 1 lognormal + midpoint fallback + anomalies → Task 4. ✓
- Path 2 median date (resolution_date-first per Revision #1) → Tasks 5–6. ✓
- Path 3 argmax → Tasks 5–6. ✓
- Path 4 boolean median + `heterogeneous_group` → Tasks 5–6. ✓
- `PointEstimate` / `OverviewSummary` → Task 3. ✓
- `synthesize_overview` on protocol + both providers + fallback templates → Task 8. ✓
- `POST /summarize/{run_id}`, `GET /aggregate/{run_id}/{object_id}` → Task 10. ✓
- Continuous/threshold objects routed to Path 1 (Revision #3) → Task 6. ✓
- Parser consolidation (Revision #2) → Task 7. ✓
- No-scipy lognormal (Revision #6) → Task 4 uses `statistics.NormalDist`. ✓
- Frontend JSON + fetch (chosen decision) → Tasks 10–11. ✓

**2. Placeholder scan:** No "TBD"/"add error handling"/"similar to Task N". Each code step shows full code. ✓

**3. Type consistency:**
- `Interval(lower, upper, probability)` defined in Task 4, used identically in Task 6. ✓
- `numeric_estimate(intervals, unit) -> NumericEstimate` (Task 4) consumed by `_numeric_to_point_estimate` (Task 6). ✓
- `compute_point_estimate(object_name, object_type, markets)` defined Task 6, called in `service.summarize` Task 9. ✓
- `describe_point_estimates` / `_ordered_by_role` defined Task 8, imported by fallback + deepseek same task. ✓
- `SearchRunRepository.get_topic` defined Task 9, called same task. ✓
- `PointEstimate.object_id` defined Task 3, stamped in `compute_point_estimate` (Task 6), consumed by `app.js` (Task 11). ✓

**4. Review fixes folded in (2026-06-13):**
- Task 1 final branch now requires a `$`/unit cue, so the Step 3 code passes its own Step 4 test (`"No IPO by December 31, 2027"` → `None`) without the prior "fails-then-patch" note. ✓
- `describe_point_estimates` always emits one line per estimate, so anomalies never index an empty list or attach to the wrong card (reachable via a TIME object with no resolvable date — would have 500'd the DeepSeek `/summarize` path). ✓
- `PointEstimate.object_id` added end-to-end to remove the summarize-vs-aggregate key ambiguity for objects that share a display name. ✓
- Numeric path raises `residual_mass_dropped` when excluded non-numeric buckets exceed 5% mass, instead of silently dropping them. ✓
- `_numeric_from_brackets` typed as `list[tuple[MoneyBracket, float]]` (no `# type: ignore`); `np.polyfit` result indexed rather than unpacked for mypy --strict. ✓

**Known follow-ups (out of scope, flagged for later):** Open Question #10 (boolean-from-`No IPO`-residual) is left as a separate object assumption — revisit if real data lacks a distinct boolean object. `chart_json` dead output (#7) deferred to a cleanup PR. `OverviewSummary` is recomputed (and re-calls the AI) on every `/summarize` POST — persist + cache it in a follow-up (see Task 10 note).
</content>
</invoke>
