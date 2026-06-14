<div align="center">

<img src="app/web/static/favicon.svg" width="76" alt="Polymarket Overview" />

# Polymarket Overview

**Collapse a whole prediction market into one number — and one sentence.**

Search any topic → pull the live Polymarket markets → structure every question →
synthesize each data object into a single **point estimate** plus one AI narrative line,
with the full probability distribution one click away.

[![Python](https://img.shields.io/badge/python-3.12+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Pydantic v2](https://img.shields.io/badge/Pydantic-v2-E92063?logo=pydantic&logoColor=white)](https://docs.pydantic.dev/)
[![tests](https://img.shields.io/badge/tests-61%20passing-2ea44f)](#verify)
[![types](https://img.shields.io/badge/mypy-strict-1f6feb)](#verify)
[![style](https://img.shields.io/badge/ruff-checked-D7FF64?logo=ruff&logoColor=black)](#verify)

</div>

---

## Pipeline

```
  topic
    │
    ▼
  Polymarket API ──► BM25 rank ──► DeepSeek structure ──► repair + strict validate
                                                                   │
                                                                   ▼
                                          structured markets ──► human review (optional)
                                                                   │
                                                                   ▼
                                                         aggregate per data object
                                                                   │
              ┌────────────────────────────────────────────────────┤
              ▼                          ▼                          ▼
      point estimate            AI narrative              full distribution
       (one number)            (one sentence)              (on demand)
```

Probabilities are **never** invented by the model — they are computed deterministically
from Polymarket outcome prices. The LLM only structures markets and writes the closing
narrative.

## The four estimation paths

Each data object is routed by type to one deterministic estimator (no scipy — just
`numpy` + `statistics.NormalDist`). The four roles map to the four accent colors in the UI:

| | Role | Object type | Method | Headline |
|---|---|---|---|---|
| 🟢 | **Numeric** | `continuous` | Lognormal least-squares fit of `ln(threshold)` vs `Φ⁻¹(cdf)`, gated at **R² ≥ 0.85**; probability-weighted midpoint fallback | expected value + p10 / p50 / p90 |
| 🔵 | **Boolean** | `boolean` | Median yes-probability across the group | probability |
| 🟣 | **Time** | `time` | Median resolution date (`resolution_date`, parsed deadline as fallback) | date |
| 🔴 | **Context** | `categorical` | Arg-max of the normalized category probabilities | top category + confidence |

Data-quality **anomalies** — `monotonicity_conflict`, `high_tail_probability`,
`residual_mass_dropped`, `heterogeneous_group`, … — ride along on each estimate and are
fed to the narrative so it can hedge instead of overclaiming.

## Quickstart

```bash
uv sync --group dev
cp .env.example .env          # add your DeepSeek key (optional — falls back to rule-based)
uv run uvicorn app.main:app --port 8787
```

Open **http://127.0.0.1:8787** and search a topic — try `Anthropic IPO` or
`Bitcoin price 2025`.

> Without a DeepSeek key the app still runs end-to-end on a deterministic rule-based
> provider; set the key for real structuring and narratives.

## Configuration

All settings are environment variables (prefix `POLYMARKET_OVERVIEW_`, read from `.env`):

| Variable | Default | Purpose |
|---|---|---|
| `DEEPSEEK_API_KEY` | — | DeepSeek key; empty → rule-based fallback provider |
| `DEEPSEEK_MODEL` | `deepseek-chat` | Model for structuring + narrative |
| `GITHUB_URL` | — | When set, shows a GitHub link in the nav |
| `PORT` | `8787` | Server port |
| `DB_PATH` | `data/polymarket-overview.sqlite` | SQLite store |

## HTTP surface

| Method | Path | Returns |
|---|---|---|
| `POST` | `/search` | data objects + structured markets for a topic |
| `GET` | `/review/{run}` | editable structured-market review |
| `POST` | `/summarize/{run}` | `OverviewSummary` — point estimates + narrative · `404` if unknown |
| `GET` | `/aggregate/{run}/{object}` | `AggregationResult` — the full distribution · `404` if unknown |

## Architecture

```
app/
├── polymarket/     live market fetch + normalization
├── recall/         BM25 relevance ranking
├── ai/             provider abstraction · DeepSeek · rule-based fallback · boundary repair
├── analysis/       parsers · aggregators · point_estimate · anomalies
├── storage/        sqlite repositories
├── web/            FastAPI routes · Jinja templates · static (mono/paper UI)
└── service.py      orchestration: search → structure → aggregate → summarize
```

The AI boundary is hardened: malformed LLM output (date strings or `[lo, hi]` ranges in
the numeric `threshold_value` field) is repaired to the `number|null` contract before
strict Pydantic validation, so one bad market never 500s a run — while genuine contract
violations (e.g. injected probabilities) still fail loudly.

## Verify

```bash
uv run pytest -q          # 61 passing
uv run ruff check app tests
uv run mypy app           # strict, clean
```

CLI (no browser):

```bash
uv run python -m app.cli "Anthropic IPO" --limit 60 --summary
```

---

<div align="center">
<sub>Built with FastAPI · Pydantic v2 · DeepSeek · a little lognormal algebra.</sub>
</div>
