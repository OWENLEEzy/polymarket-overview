# Point Estimate Synthesis Design

**Date:** 2026-06-13  
**Status:** Approved

## Problem

The system currently outputs a probability distribution per market object — multiple rows of numbers. The goal is to produce a single point estimate per object and one AI-generated narrative sentence for the whole topic.

## Goals

1. Each market object produces one point estimate (a number, a date, or a category).
2. All point estimates feed into one AI-generated narrative sentence.
3. The full distribution remains available as a collapsed detail view (click to expand).

---

## Data Flow

```
POST /search
    → StructuredMarkets saved to DB

POST /summarize/{run_id}
    → Load StructuredMarkets from DB
    → Group by object_id
    → For each object:
        a. aggregate()              → AggregationResult  (stored in detail{})
        b. compute_point_estimate() → PointEstimate
    → ai.synthesize_overview(topic, point_estimates) → narrative
    → Return OverviewSummary

GET /aggregate/{run_id}/{object_id}   (on demand, for detail view)
    → Return AggregationResult
```

---

## Input Taxonomy

Real Polymarket data uses two operators: `category` and `=`. All numeric values and dates are embedded in `category_value` strings. The system routes each object through one of four paths based on object type and whether `category_value` is parseable.

| Object type | Category value | Path |
|-------------|---------------|------|
| `categorical` / `time` | Contains numeric range (`"$1.25–$1.5T"`) | Path 1: numeric bracket |
| `time` | Contains date phrase (`"IPO by Oct 31, 2026"`) | Path 2: date |
| `categorical` | Non-numeric text (`"No IPO"`, `"Anthropic"`) | Path 3: argmax |
| `boolean` | `"yes"` / `"no"` | Path 4: median |

---

## Parsing Layer (`analysis/parsers.py`)

### Numeric bracket parser

Handles these patterns:

```
"$1.25–$1.5T"  → (lower=1.25, upper=1.50, unit="T")
"$3.0T+"       → (lower=3.0,  upper=None, unit="T")
"<$1.25T"      → (lower=None, upper=1.25, unit="T")
"600B+"        → (lower=600,  upper=None, unit="B")
"No IPO"       → None   (skip)
```

Returns `None` for any string that does not contain a parseable number.

### Date parser

Extracts the deadline date from timing question strings:

```
"Will Anthropic IPO by October 31, 2026?" → 2026-10-31
```

Returns `None` if no date is found.

---

## Point Estimate Algorithms

### Path 1 — Numeric bracket → Expected Value

**Primary method: lognormal fitting**

1. Exclude buckets where `category_value` parses to `None` (e.g., "No IPO"). Compute conditional EV given the event occurs.
2. Normalize remaining bucket probabilities to sum to 1.
3. Build CDF points at each upper bound: `CDF(upper_i) = sum of probabilities for all buckets with upper ≤ upper_i`.
4. Transform: `y = ln(threshold)`, `z = Φ⁻¹(CDF)`.
5. Fit `y = μ + σ·z` via least squares.
6. Compute `EV = exp(μ + σ²/2)`. Also output `P10`, `P50`, `P90`.

**Fallback: midpoint weighted average**

Triggered when:
- Fewer than 3 parseable CDF points
- `σ ≤ 0` (degenerate fit)
- R² of the linear fit < 0.85
- Any threshold ≤ 0

Fallback assigns midpoints to bounded intervals (`(lower + upper) / 2`) and `lower × 1.5` to the unbounded upper tail. Flags `anomaly = "lognormal_fit_failed"`.

**Anomalies detected:**
- `lognormal_fit_failed` — fallback to midpoint
- `high_tail_probability` — unbounded tail bucket > 30% (midpoint EV unreliable)
- `insufficient_data` — fewer than 2 included markets

### Path 2 — Date strings → Median date

1. Parse each `category_value` to extract a deadline date.
2. Sort by date ascending. These form a CDF: `P(event by date_i)`.
3. Return the first date where cumulative probability ≥ 50%.
4. If no date exceeds 50%, flag `anomaly = "median_date_not_reached"` and return the last available date.

### Path 3 — Text categorical → Argmax

1. Group markets by `category_value`, average probabilities per group.
2. Normalize across groups.
3. Return the category with the highest normalized probability and its probability.

### Path 4 — Boolean → Median probability

1. Collect probabilities from all included markets in the object.
2. Return the median (robust to outliers from misclassified markets).
3. If standard deviation > 0.30, flag `anomaly = "heterogeneous_group"`.

---

## Data Models

### New: `PointEstimate`

```python
class PointEstimate(BaseModel):
    object_name: str
    object_type: ObjectType
    role: Literal["boolean", "time", "numeric", "context"]

    # boolean
    boolean_probability: float | None = None

    # time
    median_date: str | None = None          # e.g. "2026-10-31"

    # numeric (lognormal output)
    expected_value: float | None = None
    p10: float | None = None
    p50: float | None = None
    p90: float | None = None
    unit: str | None = None                 # "T", "B", etc.

    # categorical
    top_category: str | None = None
    top_category_probability: float | None = None

    anomalies: list[str]
    fit_method: Literal["lognormal", "midpoint", "argmax", "median"]
```

### New: `OverviewSummary`

```python
class OverviewSummary(BaseModel):
    search_run_id: str
    topic: str
    narrative: str                              # AI-generated sentence
    point_estimates: list[PointEstimate]
```

The frontend fetches distribution data on demand via `GET /aggregate/{run_id}/{object_id}` when the user expands a detail section. `AggregationResult` is unchanged.

---

## AI Narrative Synthesis

### Role priority

The narrative follows this order:

1. **boolean** — does the event happen? (leads the sentence)
2. **time** — when? (follows boolean)
3. **numeric** — how much? (conditional on boolean = yes)
4. **context** — comparison or ranking (included if space allows)

### Prompt

**System:** You are a prediction market analyst. Distill market data into one or two sentences. Use conditional language when appropriate ("if X happens…"). Flag high uncertainty when anomalies are present. Give the conclusion directly — do not explain the method.

**User:**
```
Topic: {topic}
Point estimates (ordered by importance):
{formatted point_estimates}

Summarize the market's view in one or two sentences.
```

**Example output:**
> "市场预测 Anthropic 有 75% 的概率 IPO，最可能在 2026 年 10 月；若成功，预期市值约 $1.6T（2026 年底），但 $3T+ 区间不确定性较高。"

### Fallback (RuleBasedProvider)

Template-based assembly when AI is unavailable:

```
boolean only:
  "市场预测{topic}发生概率为{pct}%。"

boolean + time:
  "市场预测{topic}发生概率为{pct}%，最可能在{date}。"

boolean + time + numeric:
  "市场预测{topic}发生概率为{pct}%，最可能在{date}；
   若发生，预期{object_name}约{value}{unit}。"

no boolean, categorical only:
  "市场认为{topic}最可能的结果是{top_category}（{pct}%概率）。"
```

### AIProvider interface (new method)

```python
class AIProvider(Protocol):
    async def synthesize_overview(
        self,
        topic: str,
        point_estimates: list[PointEstimate],
    ) -> str: ...
```

Both `DeepSeekProvider` and `RuleBasedProvider` implement this method.

---

## Web Routes

| Method | Path | Returns | Notes |
|--------|------|---------|-------|
| `POST` | `/search` | `SearchResult` | Unchanged |
| `POST` | `/summarize/{run_id}` | `OverviewSummary` | New |
| `GET` | `/aggregate/{run_id}/{object_id}` | `AggregationResult` | Kept for detail view |

---

## File Changeset

| File | Action | Purpose |
|------|--------|---------|
| `app/analysis/parsers.py` | Create | Bracket and date string parsing |
| `app/analysis/point_estimate.py` | Create | Four-path point estimate computation |
| `app/analysis/aggregators.py` | Keep | Supplies detail view data |
| `app/models.py` | Modify | Add `PointEstimate`, `OverviewSummary` |
| `app/ai/prompts.py` | Modify | Add narrative synthesis prompt |
| `app/ai/provider.py` | Modify | Add `synthesize_overview()` to protocol |
| `app/ai/deepseek.py` | Modify | Implement `synthesize_overview()` |
| `app/ai/fallback.py` | Modify | Implement template-based fallback |
| `app/service.py` | Modify | Add `summarize()` method |
| `app/web/routes.py` | Modify | Add `POST /summarize/{run_id}` |
| `tests/test_parsers.py` | Create | Unit tests for parsers |
| `tests/test_point_estimate.py` | Create | Unit tests for all four paths |
| `tests/test_web_routes.py` | Modify | Add summarize endpoint tests |

---

## Open Questions & Revisions (review 2026-06-13)

Raised during a repo cross-check before implementation. Resolve before coding.

### High

1. **Path 2 should use `resolution_date`, not parse question text.**
   Every `StructuredMarket` already carries an ISO `resolution_date` (sourced from
   `RawMarket.end_date`). Parsing `"...by October 31, 2026?"` from the question is
   redundant and fragile. Prefer `resolution_date`; only fall back to text parsing
   when it is missing. The date parser in the spec drops to a fallback role.

2. **Path 1 parser duplicates existing code — consolidate, don't fork.**
   `analysis/aggregators.py` already parses money strings via `normalize_money_value()`
   and `category_sort_key()` (handles `$1.25T`, `<$1.25T`, `No IPO`; see
   `test_categorical_money_buckets_sort_and_normalize`). The changeset marks
   `aggregators.py` "Keep", which leaves two money parsers. Extract the shared parser
   into `analysis/parsers.py` and have `aggregators.py` import it.

3. **`ObjectType.CONTINUOUS` (threshold-based) has no path in the taxonomy.**
   The four-path table only routes objects whose numbers live in `category_value`
   strings. But `continuous` objects carry their number in `threshold_value` with a
   real operator (`>`, `<`, …) — these are produced by `RuleBasedProvider` (fallback
   mode, no API key) and consumed by `aggregate_continuous()` (tested). The spec's
   claim "real data uses only `category` and `=`" holds only for the AI/bucketed
   representation; the threshold representation still exists. Either: (a) route
   `continuous` + threshold objects into Path 1 by deriving CDF points from
   `threshold_value` + `operator` (reuse `threshold_cdf_probability`), or (b) state
   explicitly that point estimates are out of scope for that representation. Today they
   would fall through with no estimate.

### Medium

4. **Specify the `object → (path, role)` decision function.**
   `ObjectType` is `continuous/categorical/boolean/time`; `PointEstimate.role` is
   `boolean/time/numeric/context`. The rule that sends a `categorical` object to
   Path 1 (numeric) vs Path 3 (argmax/context) — i.e. "does `category_value` parse to
   a number?" — must be an explicit function with tests, including **mixed buckets**
   (money buckets + a `"No IPO"` bucket in the same object).

5. **Path 1 must unify units before fitting.**
   Buckets may mix units (`"600B+"` vs `"$1.5T"`). Convert all thresholds to one
   canonical unit (reuse `normalize_money_value`, T→B) before the lognormal fit; set
   `PointEstimate.unit` to that canonical unit.

6. **Lognormal needs no scipy.** Use `statistics.NormalDist().inv_cdf` for Φ⁻¹ and
   `numpy.polyfit` for the least-squares fit (numpy already a dependency). Do not add scipy.

7. **Frontend changeset is incomplete + implies an architecture shift.**
   The app is currently pure server-rendered Jinja with zero JS (verified: no
   `script`/`fetch`/`plotly` reference under `app/web/`). Switching to a JSON
   `POST /summarize` + on-demand `GET /aggregate` + client-side expand adds a fetch
   layer not listed in the changeset (templates, `app.css`, client JS all missing).
   **Recommended simpler path:** render the detail view as a server-side `<details>`
   collapse instead of a JSON API. Also decide the fate of the existing
   `POST /overview` route and `overview.html`.
   Note: `AggregationResult.chart_json` is already computed by `aggregators.py` but
   **rendered by no template today** (dead output). If the detail view stays
   server-side, either wire it up or stop computing it; don't carry it forward unused.

### Low

8. `AIProvider` is an `ABC` with `@abstractmethod`, not a `Protocol` — align wording.
9. `synthesize_overview()` returns `str`, but `DeepSeekProvider` only has `_chat_json`
   (returns dict). Add a `_chat_text` helper.
10. **Boolean source:** real market-cap data often includes a `"No IPO"` bucket, so
   `P(IPO)` may be the residual `1 − P(No IPO)` rather than a separate boolean object.
   Confirm the data yields a distinct boolean object; otherwise derive it from the residual.

### Infra fix applied

- `pyproject.toml` had no `pythonpath`, so `uv run pytest` failed with
  `No module named 'app'`. Added `pythonpath = ["."]` under `[tool.pytest.ini_options]`.
  (The local `.venv` was also stale — pointed at a pre-move path — and was rebuilt with
  `uv sync`.)
