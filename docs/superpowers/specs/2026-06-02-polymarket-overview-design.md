# Polymarket Overview Design

Date: 2026-06-02

Status: approved for implementation planning

Project path:

```text
/Users/owenlee/Desktop/polymarket-overview
```

## 1. Goal

Build a local web tool that turns related Polymarket markets into a macro overview.

The first version focuses on data organization and deterministic analysis. The user enters a topic, the system searches Polymarket in real time, AI helps identify and structure related markets, the user confirms the target data object, and Python code aggregates the confirmed markets into tables and simple charts.

The tool does not trade, place orders, or treat AI-generated probabilities as facts.

## 2. Product Scope

### In Scope

- Search Polymarket by topic on demand.
- Use AI to plan search terms, judge relevance, extract structured fields, and group markets by data object.
- Let the user select a target data object.
- Let the user edit structured fields before aggregation.
- Aggregate continuous, categorical, boolean, and time-like data objects with deterministic Python code.
- Save raw API responses, AI outputs, user corrections, and aggregation results in a local database.
- Show overview tables first and charts second.

### Out Of Scope

- Full local indexing of all Polymarket markets.
- Multi-user accounts or permissions.
- Remote deployment.
- Automatic trading or arbitrage execution.
- Paid data sources.
- AI-generated final probability calculations.

## 3. User Flow

```text
User enters topic
  -> DeepSeek proposes search terms
  -> Polymarket API returns candidate events and markets
  -> Local recall ranks candidates with BM25 and embeddings
  -> DeepSeek structures markets into data objects
  -> User selects a data object
  -> User edits market fields and inclusion flags
  -> Python aggregation code computes the overview
  -> UI shows tables, simple charts, anomalies, and audit records
```

Example topic:

```text
Anthropic IPO
```

Example extracted data objects:

- Anthropic IPO occurrence
- Anthropic IPO valuation
- Anthropic IPO timing
- Anthropic IPO exchange or listing route

## 4. Pages

| Page | Purpose |
| --- | --- |
| Search | Topic input and candidate Polymarket markets. |
| Objects | AI-extracted data objects. The user chooses the target object. |
| Review | Editable structured fields for each market. |
| Overview | Aggregation tables, charts, anomalies, and audit links. |

The UI should stay simple. The product value is retrieval, structure extraction, and analysis, not frontend complexity.

## 5. Data Sources

Primary source:

- Polymarket Gamma API for events, markets, outcomes, metadata, and discovery.

Price source:

- Polymarket CLOB API public endpoints for midpoint, spread, and orderbook data when token IDs are available.

The design uses public read endpoints only. It does not use authenticated trading endpoints.

Reference facts from official Polymarket docs:

- Gamma API is the primary discovery API for markets and events.
- CLOB API exposes public orderbook, price, midpoint, spread, and history endpoints.
- Polymarket prices represent implied probabilities.
- Displayed price is generally bid-ask midpoint, with last trade used when spread is wide.

References:

- https://docs.polymarket.com/quickstart/reference/endpoints
- https://docs.polymarket.com/developers/gamma-markets-api/overview
- https://docs.polymarket.com/trading/orderbook
- https://docs.polymarket.com/api-reference/data/get-midpoint-price

## 6. AI Boundary

AI is a structuring assistant, not a calculator.

AI may:

- Rewrite the user topic into search queries.
- Judge whether a market belongs to the topic.
- Extract data objects.
- Classify object type.
- Extract thresholds, categories, units, operators, and resolution dates.
- Explain uncertainty and grouping decisions.

AI may not:

- Invent market data.
- Replace Polymarket API facts.
- Compute final probability distributions.
- Hide low-confidence assumptions.

All AI outputs must pass Pydantic validation before the app stores or uses them.

## 7. Technical Stack

| Layer | Choice |
| --- | --- |
| Backend | Python + FastAPI |
| Pages | Jinja2 + HTMX |
| HTTP | httpx |
| Schemas | Pydantic |
| Database | SQLite first; DuckDB optional for analytical tables |
| Keyword recall | rank-bm25 |
| Semantic recall | sentence-transformers |
| AI provider | Provider interface, DeepSeek as default demo provider |
| Analysis | pandas, numpy, scikit-learn |
| Charts | Plotly |
| Tests | pytest |

Avoid large agent frameworks in v1. Use existing focused packages and keep the workflow explicit.

## 8. Local Database Model

First version tables:

| Table | Purpose |
| --- | --- |
| search_runs | One row per user topic search. |
| raw_markets | Raw Polymarket event, market, outcome, and price JSON. |
| structured_markets | AI-extracted structured fields for each market. |
| aggregation_runs | User-confirmed inputs, output tables, chart data, and anomaly records. |

Each run should preserve:

- Original topic.
- Search terms.
- Raw API payloads.
- AI provider and model name.
- Structured AI JSON.
- User edits.
- Aggregation result.
- Created timestamp.

## 9. Structured Market Fields

Minimum fields:

```text
market_id
question
event_title
object_name
object_type: continuous | categorical | boolean | time
operator: > | >= | < | <= | = | range | category
threshold_value
threshold_unit
category_value
probability
probability_source: midpoint | last_price | displayed_price | outcome_price
resolution_date
include
confidence
explanation
```

The Review page must let the user edit these fields before aggregation.

## 10. Aggregation Rules

### Continuous Objects

Convert threshold markets into a CDF or survival curve, then compute interval probabilities.

Example:

```text
P(X > 85B) = 0.70
P(X > 100B) = 0.45
P(X > 150B) = 0.10

P(X <= 85B) = 0.30
P(85B < X <= 100B) = 0.25
P(100B < X <= 150B) = 0.35
P(X > 150B) = 0.10
```

If thresholds violate monotonicity, flag the issue and optionally show a corrected curve using isotonic regression.

### Time Objects

Treat time as either:

- A continuous variable when markets use before/after thresholds.
- A bucketed categorical variable when markets use periods such as Q1, 2027, or before 2028.

Do not merge incompatible time definitions without user confirmation.

### Categorical Objects

Group by category. If categories are mutually exclusive and cover the relevant outcome space, show normalized distribution. If not, show raw probabilities and mark the set as non-exclusive.

### Boolean Objects

For equivalent yes/no markets, compute a weighted average and consistency checks. Show outliers rather than hiding them.

## 11. Anomaly Checks

| Check | Example | Behavior |
| --- | --- | --- |
| Monotonicity conflict | `P(>100B)` exceeds `P(>85B)` | Flag and offer corrected curve. |
| Definition conflict | post-money valuation vs market cap | Do not auto-merge. |
| Date conflict | different resolution dates | Down-rank or require user confirmation. |
| Liquidity risk | wide bid-ask spread | Mark as low confidence. |
| Non-exclusive categories | before 2027 and in 2026 | Do not normalize as one distribution. |

## 12. Price Selection

Preferred probability source:

```text
midpoint or bid-ask midpoint
  > last trade price
  > displayed price or Gamma outcome price
```

If orderbook data is unavailable, use the best available Polymarket API price and label the source.

## 13. Provider Interface

DeepSeek is the default demo provider.

The code should expose an AI provider interface so additional providers can be added without changing aggregation logic:

- DeepSeek
- OpenAI
- Anthropic
- Ollama or another local model

The provider layer returns validated structured JSON only. It should not return final overview tables.

## 14. Acceptance Criteria

| Scenario | Pass condition |
| --- | --- |
| Topic search | `Anthropic IPO` returns candidate Polymarket markets and stores raw results. |
| Object extraction | AI produces at least two candidate data objects with explanations. |
| User correction | User can edit structured fields and save the corrections. |
| Continuous aggregation | Threshold markets produce an interval probability table and curve data. |
| Categorical or boolean aggregation | App displays distribution or consistency checks without unsafe normalization. |
| Audit trail | User can inspect raw API JSON, AI JSON, user edits, and final result. |
| Tests | Aggregation functions and Pydantic validation have pytest coverage. |

## 15. Implementation Notes

Keep the project independent from T46 apps. Do not integrate with `t46-pm-agent` in v1.

Expected source tree after implementation planning:

```text
polymarket-overview/
  docs/superpowers/specs/
  app/
    api/
    ai/
    analysis/
    polymarket/
    recall/
    storage/
    web/
  tests/
  pyproject.toml
```

The next step is to write a detailed implementation plan. Do not implement code until the spec is reviewed.
