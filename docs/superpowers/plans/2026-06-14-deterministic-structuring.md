# Deterministic Structuring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the LLM market-structuring call with a deterministic structurer driven by Polymarket's own fields + existing parsers, eliminating the token-truncation crash by construction.

**Architecture:** A new pure module `app/analysis/structurer.py` groups raw markets by `event_id`, classifies each group's `object_type`, and derives operator/threshold/date/category from `group_item_title`/`group_item_range`/`end_date`/`question` via existing parsers (with `python-dateutil` for dates). `service.search` calls it instead of `ai_provider.structure_markets` + `enrich_extraction`. All LLM structuring code is then deleted. The LLM keeps only `plan_search_terms` and `synthesize_overview`.

**Tech Stack:** Python 3.12, Pydantic v2, `python-dateutil`, existing `parse_money_bracket`/`canonical_unit`, `numpy`. Tooling: `uv run pytest -q`, `uv run ruff check`, `uv run mypy app`.

**Spec:** `docs/superpowers/specs/2026-06-14-deterministic-structuring-design.md`

---

## File Structure

- **Create** `app/analysis/structurer.py` — the single source of truth for structuring. `structure_markets(topic, raw_markets) -> StructuredExtraction`.
- **Create** `tests/test_structurer.py` — TDD coverage for the structurer.
- **Modify** `app/analysis/parsers.py` — `parse_deadline_date` delegates to `dateutil`.
- **Modify** `tests/test_parsers.py` — extend date-parse coverage.
- **Modify** `app/service.py` — call the structurer; drop `enrich_extraction` import.
- **Modify** `app/ai/provider.py` — delete `structure_markets` abstract method + all repair helpers.
- **Modify** `app/ai/deepseek.py` — delete `structure_markets`/`_structure_batch`/batching.
- **Modify** `app/ai/fallback.py` — delete `structure_markets` + its helpers.
- **Modify** `app/models.py` — delete `StructuredMarketDraft`, `StructuredExtractionDraft`.
- **Modify** `app/ai/enrichment.py` — keep `probability_from_outcomes`, `build_objects`, `build_structured_market`; delete `enrich_extraction` (logic absorbed by structurer).
- **Delete** `tests/test_deepseek_provider.py`.
- **Modify** `tests/test_ai_provider.py` — delete structuring tests; keep narrative/fallback tests.
- **Modify** `pyproject.toml` — add `python-dateutil>=2.9`.

---

## Task 1: Swap hand-rolled date regex for `python-dateutil`

**Files:**
- Modify: `pyproject.toml:6-19` (dependencies list)
- Modify: `app/analysis/parsers.py:68-96`
- Test: `tests/test_parsers.py`

- [ ] **Step 1: Add the dependency**

In `pyproject.toml`, add to the `dependencies` array (after `"numpy>=1.26.0",`):

```toml
  "python-dateutil>=2.9.0",
```

Run: `uv sync` — Expected: resolves and installs `python-dateutil`.

- [ ] **Step 2: Write failing tests for broader date formats**

Add to `tests/test_parsers.py`:

```python
from app.analysis.parsers import parse_deadline_date


def test_parse_deadline_iso_format() -> None:
    assert parse_deadline_date("Resolves 2026-06-30") == "2026-06-30"


def test_parse_deadline_day_month_year() -> None:
    assert parse_deadline_date("by 30 June 2026") == "2026-06-30"


def test_parse_deadline_month_name_format() -> None:
    assert parse_deadline_date("No IPO by December 31, 2027") == "2027-12-31"


def test_parse_deadline_no_date_returns_none() -> None:
    assert parse_deadline_date("Will Anthropic IPO?") is None


def test_parse_deadline_empty_returns_none() -> None:
    assert parse_deadline_date(None) is None
    assert parse_deadline_date("") is None
```

- [ ] **Step 3: Run tests to verify the ISO/day-first ones fail**

Run: `uv run pytest tests/test_parsers.py -k deadline -v`
Expected: `test_parse_deadline_iso_format` and `test_parse_deadline_day_month_year` FAIL (old regex only matches `Month DD, YYYY`).

- [ ] **Step 4: Reimplement `parse_deadline_date` with dateutil**

In `app/analysis/parsers.py`, delete the `_MONTHS`, `_DATE_RE` constants (lines 68-82) and the body of `parse_deadline_date` (lines 85-96). Replace with:

```python
from dateutil import parser as _dateparser


def parse_deadline_date(text: str | None) -> str | None:
    if not text or not text.strip():
        return None
    try:
        parsed = _dateparser.parse(text, fuzzy=True, default=None)
    except (ValueError, OverflowError, TypeError):
        return None
    if parsed is None:
        return None
    return f"{parsed.year:04d}-{parsed.month:02d}-{parsed.day:02d}"
```

Add `from dateutil import parser as _dateparser` to the imports at the top of the file (keep the existing `import re` — `parse_money_bracket` still needs it).

- [ ] **Step 5: Run the full parser suite**

Run: `uv run pytest tests/test_parsers.py -v`
Expected: ALL pass, including the 5 new deadline tests and all existing money-bracket tests.

> Note: `fuzzy=True` with `default=None` returns `None` when no date tokens are found, instead of defaulting to today. Verify `test_parse_deadline_no_date_returns_none` passes — this guard is load-bearing.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock app/analysis/parsers.py tests/test_parsers.py
git commit -m "refactor: parse deadline dates with python-dateutil"
```

---

## Task 2: Structurer — group by event_id, assign one object_id per group

**Files:**
- Create: `app/analysis/structurer.py`
- Test: `tests/test_structurer.py`

This task builds grouping only; classification comes in Task 3.

- [ ] **Step 1: Write the failing grouping test**

Create `tests/test_structurer.py`:

```python
from app.analysis.structurer import group_markets
from app.models import RawMarket


def _raw(market_id: str, event_id: str | None, **kw: object) -> RawMarket:
    base: dict[str, object] = {
        "market_id": market_id,
        "event_id": event_id,
        "question": kw.get("question", f"q {market_id}"),
        "event_title": kw.get("event_title", "Event"),
        "outcomes": kw.get("outcomes", ["Yes", "No"]),
        "outcome_prices": kw.get("outcome_prices", [0.5, 0.5]),
        "raw_json": {"id": market_id},
    }
    base.update({k: v for k, v in kw.items() if k in RawMarket.model_fields})
    return RawMarket.model_validate(base)


def test_group_markets_groups_by_event_id() -> None:
    markets = [
        _raw("m1", "e1"),
        _raw("m2", "e1"),
        _raw("m3", "e2"),
    ]
    groups = group_markets(markets)
    keys = sorted(len(members) for members in groups.values())
    assert keys == [1, 2]


def test_group_markets_falls_back_to_event_title_then_market_id() -> None:
    markets = [
        _raw("m1", None, event_title="Shared"),
        _raw("m2", None, event_title="Shared"),
        _raw("m3", None, event_title=None),
    ]
    groups = group_markets(markets)
    sizes = sorted(len(members) for members in groups.values())
    assert sizes == [1, 2]
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_structurer.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.analysis.structurer'`.

- [ ] **Step 3: Implement `group_markets`**

Create `app/analysis/structurer.py`:

```python
from collections import defaultdict

from app.models import RawMarket


def _group_key(market: RawMarket) -> str:
    if market.event_id:
        return f"event:{market.event_id}"
    if market.event_title:
        return f"title:{market.event_title.strip().casefold()}"
    return f"market:{market.market_id}"


def group_markets(markets: list[RawMarket]) -> dict[str, list[RawMarket]]:
    groups: dict[str, list[RawMarket]] = defaultdict(list)
    for market in markets:
        groups[_group_key(market)].append(market)
    return dict(groups)
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_structurer.py -v`
Expected: both tests PASS.

- [ ] **Step 5: Commit**

```bash
git add app/analysis/structurer.py tests/test_structurer.py
git commit -m "feat: group raw markets by event_id in structurer"
```

---

## Task 3: Structurer — classify object_type and derive per-market fields

**Files:**
- Modify: `app/analysis/structurer.py`
- Test: `tests/test_structurer.py`

- [ ] **Step 1: Write failing classification tests**

Add to `tests/test_structurer.py`:

```python
from app.analysis.structurer import classify_group
from app.models import ObjectType, Operator


def test_classify_continuous_ladder() -> None:
    markets = [
        _raw("m1", "e1", group_item_title=">$100B", question="Above $100B?"),
        _raw("m2", "e1", group_item_title=">$150B", question="Above $150B?"),
    ]
    object_type, members = classify_group("Anthropic", markets)
    assert object_type == ObjectType.CONTINUOUS
    assert {m.operator for m in members} == {Operator.GREATER_THAN_OR_EQUAL}
    assert sorted(m.threshold_value for m in members) == [100.0, 150.0]


def test_classify_categorical_candidates() -> None:
    markets = [
        _raw("m1", "e1", group_item_title="Marco Rubio", question="Rubio nominee?"),
        _raw("m2", "e1", group_item_title="Nikki Haley", question="Haley nominee?"),
    ]
    object_type, members = classify_group("2028 GOP", markets)
    assert object_type == ObjectType.CATEGORICAL
    assert {m.category_value for m in members} == {"Marco Rubio", "Nikki Haley"}
    assert all(m.operator == Operator.CATEGORY for m in members)


def test_classify_time_ladder() -> None:
    markets = [
        _raw("m1", "e1", group_item_title="June 2026", question="IPO by 30 June 2026?",
             end_date="2026-06-30T00:00:00Z"),
        _raw("m2", "e1", group_item_title="December 2026", question="IPO by 31 December 2026?",
             end_date="2026-12-31T00:00:00Z"),
    ]
    object_type, members = classify_group("Anthropic IPO", markets)
    assert object_type == ObjectType.TIME
    assert {m.resolution_date for m in members} == {"2026-06-30", "2026-12-31"}


def test_classify_lone_boolean() -> None:
    markets = [_raw("m1", "e1", question="Will a recession happen in 2026?")]
    object_type, members = classify_group("Recession", markets)
    assert object_type == ObjectType.BOOLEAN
    assert members[0].operator == Operator.EQUAL
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_structurer.py -k classify -v`
Expected: FAIL — `classify_group` not defined.

- [ ] **Step 3: Implement classification + per-market derivation**

Append to `app/analysis/structurer.py`. Add imports at top:

```python
import re
from dataclasses import dataclass

from app.analysis.parsers import MoneyBracket, canonical_unit, parse_deadline_date, parse_money_bracket
from app.models import ObjectType, Operator, RawMarket
```

Then:

```python
@dataclass(frozen=True)
class ClassifiedMarket:
    raw: RawMarket
    operator: Operator
    threshold_value: float | None
    threshold_unit: str | None
    category_value: str | None
    resolution_date: str | None


def _bracket_of(market: RawMarket) -> MoneyBracket | None:
    return parse_money_bracket(market.group_item_title) or parse_money_bracket(market.question)


def _date_of(market: RawMarket) -> str | None:
    if market.end_date:
        return market.end_date.split("T")[0]
    return parse_deadline_date(market.question)


def _looks_like_date_bucket(market: RawMarket) -> bool:
    # Require a 4-digit year so fuzzy parsing does not misread a candidate name
    # (e.g. "Marco Rubio") as a date and wrongly route a categorical group to TIME.
    title = market.group_item_title or ""
    return bool(re.search(r"\b\d{4}\b", title)) and parse_deadline_date(title) is not None


def _bracket_operator(bracket: MoneyBracket) -> tuple[Operator, float]:
    # Open-upper ">=100B" => threshold is the lower edge, operator >=.
    if bracket.lower is not None and bracket.upper is None:
        return Operator.GREATER_THAN_OR_EQUAL, bracket.lower
    # Open-lower "<100B" => threshold is the upper edge, operator <.
    if bracket.lower is None and bracket.upper is not None:
        return Operator.LESS_THAN, bracket.upper
    # Closed range "100B-150B" => keep the lower edge as the representative point.
    if bracket.lower is not None and bracket.upper is not None and bracket.lower != bracket.upper:
        return Operator.RANGE, bracket.lower
    # Single "$100B".
    value = bracket.lower if bracket.lower is not None else bracket.upper
    return Operator.GREATER_THAN_OR_EQUAL, float(value or 0.0)


def classify_group(
    topic: str, markets: list[RawMarket]
) -> tuple[ObjectType, list[ClassifiedMarket]]:
    brackets = {m.market_id: _bracket_of(m) for m in markets}
    numeric = [m for m in markets if brackets[m.market_id] is not None]
    date_bucketed = [m for m in markets if _looks_like_date_bucket(m)]
    named = [
        m
        for m in markets
        if m.group_item_title and brackets[m.market_id] is None and not _looks_like_date_bucket(m)
    ]

    if len(numeric) >= 2 or (numeric and len(markets) == len(numeric)):
        object_type = ObjectType.CONTINUOUS
    elif len(date_bucketed) >= 2:
        object_type = ObjectType.TIME
    elif len(named) >= 2 or any(len(m.outcomes) > 2 for m in markets):
        object_type = ObjectType.CATEGORICAL
    else:
        object_type = ObjectType.BOOLEAN

    classified: list[ClassifiedMarket] = []
    for market in markets:
        classified.append(_classify_member(object_type, market, brackets[market.market_id]))
    return object_type, classified


def _classify_member(
    object_type: ObjectType, market: RawMarket, bracket: MoneyBracket | None
) -> ClassifiedMarket:
    if object_type == ObjectType.CONTINUOUS and bracket is not None:
        operator, threshold = _bracket_operator(bracket)
        return ClassifiedMarket(
            raw=market,
            operator=operator,
            threshold_value=threshold,
            threshold_unit=canonical_unit(bracket.unit),
            category_value=None,
            resolution_date=_date_of(market),
        )
    if object_type == ObjectType.TIME:
        return ClassifiedMarket(
            raw=market,
            operator=Operator.CATEGORY,
            threshold_value=None,
            threshold_unit=None,
            category_value=None,
            resolution_date=_date_of(market),
        )
    if object_type == ObjectType.CATEGORICAL:
        return ClassifiedMarket(
            raw=market,
            operator=Operator.CATEGORY,
            threshold_value=None,
            threshold_unit=None,
            category_value=market.group_item_title or market.question,
            resolution_date=_date_of(market),
        )
    return ClassifiedMarket(
        raw=market,
        operator=Operator.EQUAL,
        threshold_value=None,
        threshold_unit=None,
        category_value="yes",
        resolution_date=_date_of(market),
    )
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_structurer.py -k classify -v`
Expected: all 4 classification tests PASS.

> Note: `canonical_unit` returns `"B"`; the continuous aggregator's `to_billions` expects the short unit (`"B"`/`"T"`/`"M"`). `parse_money_bracket` already returns those short forms, so `threshold_unit` matches what `_continuous_point_estimate` (`app/analysis/point_estimate.py:283`) consumes.

- [ ] **Step 5: Commit**

```bash
git add app/analysis/structurer.py tests/test_structurer.py
git commit -m "feat: classify object_type and derive fields deterministically"
```

---

## Task 4: Structurer — assemble StructuredExtraction with probability join + flagging

**Files:**
- Modify: `app/analysis/structurer.py`
- Test: `tests/test_structurer.py`

- [ ] **Step 1: Write the failing assembly test**

Add to `tests/test_structurer.py`:

```python
from app.analysis.structurer import structure_markets


def test_structure_markets_builds_objects_and_joins_probability() -> None:
    markets = [
        _raw("m1", "e1", group_item_title=">$100B", question="Above $100B?",
             outcomes=["Yes", "No"], outcome_prices=[0.4, 0.6]),
        _raw("m2", "e1", group_item_title=">$150B", question="Above $150B?",
             outcomes=["Yes", "No"], outcome_prices=[0.2, 0.8]),
    ]
    extraction = structure_markets("Anthropic", markets)

    assert len(extraction.objects) == 1
    assert extraction.objects[0].object_type == ObjectType.CONTINUOUS
    probs = sorted(m.probability for m in extraction.markets)
    assert probs == [0.2, 0.4]
    # All members share one object_id (no fragmentation across the ladder).
    assert len({m.object_id for m in extraction.markets}) == 1


def test_structure_markets_flags_unstructurable() -> None:
    # Free-text Yes/No with no bracket and no date => boolean, still included.
    markets = [_raw("m1", "e1", question="Will it rain?", group_item_title=None)]
    extraction = structure_markets("Weather", markets)
    assert extraction.markets[0].include is True


def test_structure_markets_excludes_closed_markets() -> None:
    markets = [_raw("m1", "e1", question="Resolved?", closed=True)]
    extraction = structure_markets("X", markets)
    assert extraction.markets[0].include is False
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_structurer.py -k structure_markets -v`
Expected: FAIL — `structure_markets` not defined.

- [ ] **Step 3: Implement `structure_markets`**

Append to `app/analysis/structurer.py`. Add imports:

```python
from app.ai.enrichment import build_objects, probability_from_outcomes
from app.models import StructuredExtraction, StructuredMarket, build_object_id
```

Then:

```python
def _object_name(object_type: ObjectType, markets: list[RawMarket], topic: str) -> str:
    title = next((m.event_title for m in markets if m.event_title), None)
    return title or topic


def structure_markets(topic: str, raw_markets: list[RawMarket]) -> StructuredExtraction:
    groups = group_markets(raw_markets)
    structured: list[StructuredMarket] = []

    for members in groups.values():
        object_type, classified = classify_group(topic, members)
        object_name = _object_name(object_type, members, topic)
        # One object_id per group: pass None for the per-market-varying fields so a
        # ladder of differing thresholds/dates does not fragment into singletons.
        object_id = build_object_id(object_name, object_type, members[0].event_title, None, None)

        for item in classified:
            probability = probability_from_outcomes(item.raw, item.category_value)
            if probability is None:
                continue
            structured.append(
                StructuredMarket(
                    object_id=object_id,
                    market_id=item.raw.market_id,
                    question=item.raw.question,
                    event_title=item.raw.event_title or object_name,
                    object_name=object_name,
                    object_type=object_type,
                    operator=item.operator,
                    threshold_value=item.threshold_value,
                    threshold_unit=item.threshold_unit,
                    category_value=item.category_value,
                    probability=probability,
                    probability_source="outcome_price",
                    resolution_date=item.resolution_date,
                    include=not item.raw.closed and not item.raw.archived,
                    confidence=1.0,
                    explanation=f"Deterministic: {object_type.value} via {item.operator.value}",
                )
            )

    return StructuredExtraction(
        topic=topic,
        objects=build_objects(structured),
        markets=structured,
    )
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_structurer.py -v`
Expected: all structurer tests PASS.

- [ ] **Step 5: Commit**

```bash
git add app/analysis/structurer.py tests/test_structurer.py
git commit -m "feat: assemble StructuredExtraction deterministically with probability join"
```

---

## Task 5: Wire `service.search` to the structurer

**Files:**
- Modify: `app/service.py:1-68`
- Test: `tests/test_web_routes.py` (existing integration coverage)

- [ ] **Step 1: Replace the LLM structuring call**

In `app/service.py`, change the import on line 2 from:

```python
from app.ai.enrichment import enrich_extraction
```

to:

```python
from app.analysis.structurer import structure_markets
```

In `search` (lines 45-53), replace:

```python
        draft_extraction = await self.ai_provider.structure_markets(topic, selected)
        extraction = enrich_extraction(topic, selected, draft_extraction)
```

with:

```python
        extraction = structure_markets(topic, selected)
```

- [ ] **Step 2: Run the web/route integration tests**

Run: `uv run pytest tests/test_web_routes.py tests/test_enrichment.py -v`
Expected: PASS. If `tests/test_enrichment.py` references `enrich_extraction`, note the failures — they are removed in Task 6.

- [ ] **Step 3: Commit**

```bash
git add app/service.py
git commit -m "refactor: structure markets deterministically in service.search"
```

---

## Task 6: Delete all LLM structuring code

**Files:**
- Modify: `app/ai/provider.py`, `app/ai/deepseek.py`, `app/ai/fallback.py`, `app/ai/enrichment.py`, `app/models.py`
- Delete: `tests/test_deepseek_provider.py`
- Modify: `tests/test_ai_provider.py`, `tests/test_enrichment.py`

- [ ] **Step 1: Remove `structure_markets` from the provider interface**

In `app/ai/provider.py`: delete the abstract method `structure_markets` (lines 12-16) and everything from `_THRESHOLD_OPERATORS` (line 23) to end of file (`_as_float`). Keep `AIProvider` with only `plan_search_terms` and `synthesize_overview`, and keep the imports it still needs (`ABC`, `abstractmethod`, `PointEstimate`). Remove now-unused imports (`Any`, `RawMarket`, `StructuredExtractionDraft`).

- [ ] **Step 2: Remove DeepSeek structuring**

In `app/ai/deepseek.py`: delete `structure_markets`, `_structure_batch`, `_STRUCTURE_BATCH_SIZE`, and the `import asyncio` line. Remove the `parse_structured_extraction` import and `StructuredExtractionDraft` from the models import. Keep `plan_search_terms`, `synthesize_overview`, `_chat_text`, `_chat_json` (still used by `plan_search_terms`).

- [ ] **Step 3: Remove fallback structuring**

In `app/ai/fallback.py`: delete `structure_markets` (lines 31-136) and the now-orphaned helpers `VALUATION_RE`, `normalize_unit`, `is_market_cap_bucket`, `is_timing_market`, `topic_matches_text`, `yes_probability`. Keep `RuleBasedProvider` with `plan_search_terms` + `synthesize_overview`. Remove unused imports (`re`, `DataObjectCandidate`, `ObjectType`, `Operator`, `RawMarket`, `StructuredExtractionDraft`, `StructuredMarketDraft`).

- [ ] **Step 4: Remove `enrich_extraction`, keep helpers**

In `app/ai/enrichment.py`: delete `enrich_extraction` (lines 14-40). Keep `probability_from_outcomes`, `build_structured_market`, `build_objects`. Remove unused imports (`RawMarket`, `StructuredExtractionDraft`, `StructuredMarketDraft`, `build_object_id` if now unused — verify with grep).

- [ ] **Step 5: Remove draft models**

In `app/models.py`: delete `StructuredMarketDraft` (lines 95-125) and `StructuredExtractionDraft` (lines 128-133). Keep `StructuredExtraction`, `StructuredMarket`, `DataObjectCandidate`.

- [ ] **Step 6: Delete and prune tests**

```bash
git rm tests/test_deepseek_provider.py
```

In `tests/test_ai_provider.py`: delete every test that calls `parse_structured_extraction` (the structuring tests) and the `_market`/`_payload` helpers. Keep `test_fallback_narrative_boolean_time_numeric` and `test_fallback_narrative_categorical_only`. Remove the `parse_structured_extraction` import.

In `tests/test_enrichment.py`: delete tests calling `enrich_extraction`; keep tests for `probability_from_outcomes`/`build_objects` if present.

- [ ] **Step 7: Run the full suite + static gates**

Run: `uv run pytest -q && uv run ruff check app tests && uv run mypy app`
Expected: all green, no unused-import or undefined-name errors. Fix any stray import the deletions exposed.

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "refactor: delete LLM market structuring and draft models"
```

---

## Task 7: End-to-end regression against real DeepSeek

**Files:** none (verification only).

- [ ] **Step 1: Confirm no LLM structuring path remains**

Run: `grep -rn "structure_markets\|parse_structured_extraction\|StructuredExtractionDraft" app tests`
Expected: only `app/analysis/structurer.py` (the deterministic `structure_markets`) and its tests match. No `ai_provider.structure_markets` callers, no draft references.

- [ ] **Step 2: Run the four-topic end-to-end (real API key from `.env`)**

```bash
uv run python -c "
import asyncio
from app.config import get_settings
from app.service import OverviewService
TOPICS=['Bitcoin price 2026','US recession 2026','Fed rate cut 2026','2028 Republican nominee']
async def main():
    for t in TOPICS:
        svc=OverviewService(get_settings())
        res=await svc.search(t, limit=60)
        summ=await svc.summarize(res.search_run_id)
        print(f'{t}: {len(res.raw_markets)} markets -> {len(res.extraction.objects)} objects; narrative len={len(summ.narrative)}')
asyncio.run(main())
"
```

Expected: all four print without exception; object counts > 0 for topics with markets; no `JSONDecodeError`, no truncation. The only LLM calls are `plan_search_terms` and `synthesize_overview`.

- [ ] **Step 3: Final gate**

Run: `uv run pytest -q && uv run ruff check app tests && uv run mypy app`
Expected: all green; coverage of `app/analysis/structurer.py` ≥ 80%.

- [ ] **Step 4: Finish the branch**

Use superpowers:finishing-a-development-branch to verify tests and choose merge/PR.

---

## Notes for the implementer

- `MoneyBracket` (`app/analysis/parsers.py:27`) has fields `lower: float | None`, `upper: float | None`, `unit: str`. `parse_money_bracket` returns short units (`"B"`/`"T"`/`"M"`); `canonical_unit` normalizes verbose forms to those.
- `build_object_id(object_name, object_type, event_title, resolution_date, threshold_unit)` — pass `None` for the last two in the structurer so a ladder co-groups (this also fixes the pre-existing time-object fragmentation).
- `probability_from_outcomes(market, category_value)` (`app/ai/enrichment.py:43`) returns the Yes/category price or `None`; markets with no usable price are dropped (same rule the old enrich used).
- `StructuredMarket` requires `probability_source: str` and `confidence: float` — set `"outcome_price"` and `1.0`.
