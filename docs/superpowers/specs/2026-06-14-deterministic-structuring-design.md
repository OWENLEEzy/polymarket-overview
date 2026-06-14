# Deterministic Structuring (Reliability / DRY Fix) — Design

**Status:** Approved direction, pending spec review
**Date:** 2026-06-14
**Goal:** Remove the LLM from market structuring. Derive structure deterministically from Polymarket's own fields + existing parsers, so the truncation bug is impossible by construction and there is a single source of truth.

---

## 1. Root cause this fixes

The `/search` truncation crash was a **symptom of a DRY violation**. Every structuring field the LLM emits already has a deterministic source of truth:

| Fact | Deterministic source (exists today) | Duplicated by LLM (today) |
|---|---|---|
| grouping | `event_id` (+ `event_title`/`slug`) | invents `object_name` |
| object_type | `outcomes` + `group_item_title` shape + sibling count | guesses `object_type` |
| operator / threshold / unit | `parse_money_bracket(group_item_title or question)` + `group_item_range` | guesses `threshold_value`/`unit` |
| category | `group_item_title` | guesses `category_value` |
| resolution date | `end_date` or `parse_deadline_date(question)` | guesses `resolution_date` |
| relevance | BM25 ranking | guesses `include` |
| probability | `outcome_prices` | (already deterministic in enrich) |

The LLM regenerated all of this into one call whose output grew with market count until it exceeded DeepSeek's 8192-token output cap and truncated mid-JSON. **The cure is one source of truth — the deterministic one — not a leaner duplicate.**

## 2. Architecture after the change

```
Polymarket raw ─► normalize ─► BM25 rank ─► [DETERMINISTIC STRUCTURER] ─► aggregate ─► point estimate
   (search terms expanded by LLM)                  │ event_id grouping                        │
                                                   │ parse_money_bracket                       ▼
                                                   │ parse date (dateutil)            [LLM: synthesize_overview]
                                                   │ outcomes → object_type            input = a few point estimates
                                                   └ probability_from_outcomes         (hundreds of tokens, never truncates)
```

The LLM keeps exactly two jobs, both with tiny input:
- `plan_search_terms(topic)` — semantic query expansion (improves recall; no deterministic equivalent).
- `synthesize_overview(topic, point_estimates)` — the one-sentence narrative (turns anomaly codes into prose; no deterministic equivalent).

## 3. New module: `app/analysis/structurer.py`

Pure, side-effect-free function — the single source of truth for structuring:

```python
def structure_markets(topic: str, raw_markets: list[RawMarket]) -> StructuredExtraction
```

Per-market deterministic rules:

| Output field | Rule |
|---|---|
| `object_id` | `build_object_id(...)` keyed on group identity (see grouping below) |
| grouping | primary key = `event_id`; fallback `event_title`; final fallback `slug`/`market_id` |
| `object_type` | siblings in same group: ≥2 Yes/No markets each carrying a money bracket → `continuous`; date brackets → `time`; candidate-name buckets (non-Yes/No `group_item_title`) → `categorical`; a lone Yes/No with no bracket → `boolean` |
| `operator`, `threshold_value`, `threshold_unit` | `parse_money_bracket(group_item_title or question)`; `>=`/`<=`/`range` chosen from bracket open/closed sides; `group_item_range` used when present |
| `category_value` | `group_item_title` (for categorical) |
| `resolution_date` | `end_date` if present, else `parse_deadline_date(question)` |
| `probability` | `probability_from_outcomes(raw_market, category_value)` (reused from enrichment) |
| `include` | `True` if the market was placed on an axis; `False` + anomaly otherwise (see §5) |
| `confidence` | `1.0` (deterministic; field retained for schema compatibility but no longer model-fabricated) |
| `explanation` | short deterministic string, e.g. `"event_id grouping; bracket >100B"` |

Grouping + `object_type` inference is the only genuinely new logic; everything else composes existing parsers.

## 4. Library, not wheel: date parsing

Replace the hand-rolled `_DATE_RE` in `app/analysis/parsers.py` with **`python-dateutil`** (`dateutil.parser.parse`, fuzzy=True), the de-facto standard. The current regex only matches `"Month DD, YYYY"` and misses `2026-06-30`, `30 June 2026`, etc. — unacceptable now that deterministic time-object structuring depends on it.

- Add `python-dateutil>=2.9` to `[project.dependencies]`.
- `parse_deadline_date(text) -> str | None` keeps its signature and `YYYY-MM-DD` output; internally delegates to dateutil with a guarded `try/except` returning `None` on no-date.
- Keep `parse_money_bracket` hand-rolled (domain-specific bracket/range/unit semantics; no clean library equivalent) — documented as a known intentional exception.

## 5. The one behavior change

Markets the deterministic path cannot place on an axis (no event grouping AND no parseable bracket/date, free-text Yes/No) are **flagged, not outsourced to an LLM**:
- `include = False`
- anomaly tag `unstructurable` recorded on the object/market

Rationale: a market with no parseable axis position cannot contribute meaningfully to any aggregate anyway. Adding an LLM fallback here would re-introduce a second source of truth — re-committing the original DRY violation. If too many markets fall through in practice, the DRY-consistent remedy is to **extend the deterministic parser** (the single source of truth), never to bolt on a probabilistic oracle.

## 6. Removals (bug pulled out by the root)

- `AIProvider.structure_markets` abstract method.
- `DeepSeekProvider.structure_markets`, `_structure_batch`, `_STRUCTURE_BATCH_SIZE`, the `asyncio.gather` batching.
- `parse_structured_extraction`, `_repair_market`, `_coerce_threshold`, `_as_float`, `_THRESHOLD_OPERATORS` in `app/ai/provider.py`.
- `RuleBasedProvider.structure_markets`.
- `StructuredExtractionDraft`, `StructuredMarketDraft` models (no longer any LLM draft to validate).
- `tests/test_deepseek_provider.py`; the structuring tests in `tests/test_ai_provider.py` (keep the narrative/fallback tests).
- `enrich_extraction` shrinks: grouping moves into the structurer; what remains (join probability) folds into `structure_markets`. `build_objects` stays (deterministic object roll-up).

`service.search()` calls `structure_markets(topic, selected)` instead of `ai_provider.structure_markets(...)` + `enrich_extraction(...)`.

## 7. Data flow & DB (unchanged contracts)

The SQLite store (`search_runs`, `raw_markets`, `structured_markets`, `structured_market_reviews`, `aggregation_runs`) is retained: the HTTP flow is multi-request (search → summarize → aggregate share a `run_id`) and the review flow persists human edits. `structured_markets` now holds deterministically-derived rows; the review-edit and aggregation contracts are untouched.

## 8. Error handling

- Structurer never raises on a single bad market — it flags (`include=False` + anomaly) and continues. The whole run cannot fail from one malformed market.
- `dateutil` parse wrapped in try/except → `None`.
- No network call in the structuring path → no timeout/truncation/rate-limit failure modes remain in structuring.

## 9. Testing strategy (TDD)

New `tests/test_structurer.py`, RED-first, covering with real Polymarket-shaped fixtures:
- continuous ladder (Yes/No + money brackets across siblings) → one `continuous` object with sorted thresholds.
- date ladder → `time` object; `end_date` preferred over question-parsed date.
- candidate list (categorical `group_item_title`) → `categorical` with `category_value`.
- lone Yes/No, no bracket → `boolean`.
- cross-`event_id` markets → separate objects (no fragmentation, no merge).
- unstructurable free-text market → `include=False` + `unstructurable`.
- `parse_deadline_date` via dateutil: `2026-06-30`, `30 June 2026`, `December 31, 2027`, and a no-date string → `None`.

Regression: the four-topic end-to-end (Bitcoin price / recession / Fed cut / Republican nominee) must produce point estimates with **no LLM structuring call** and no truncation path.

Gates: `uv run pytest -q`, `ruff check`, `mypy --strict` all green; coverage ≥ 80%.

## 10. Out of scope (→ Spec 2)

The internal facade object (`OverviewClient`) and Polymarket anti-corruption hardening (`normalize_market` graceful degradation, isolating upstream-shape knowledge) are a separate subsystem, designed and implemented after this lands.
