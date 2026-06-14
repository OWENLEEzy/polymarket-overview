from fastapi import APIRouter, Form, HTTPException, Request
from starlette.responses import RedirectResponse
from starlette.responses import Response
from fastapi.templating import Jinja2Templates

from app.analysis.objects import build_objects
from app.config import get_settings
from app.models import (
    AggregationResult,
    ObjectType,
    Operator,
    OverviewSummary,
    StructuredMarket,
    build_object_id,
)
from app.service import OverviewService
from app.storage.db import connect, migrate
from app.storage.repositories import StructuredMarketRepository


router = APIRouter()
templates = Jinja2Templates(directory="app/web/templates")
# Expose the repository link to every template; the nav renders it only when set.
templates.env.globals["github_url"] = get_settings().github_url


@router.get("/")
async def search_page(request: Request) -> Response:
    return templates.TemplateResponse(request, "search.html", {"title": "Polymarket Overview"})


@router.post("/search")
async def search_submit(request: Request, topic: str = Form(...)) -> Response:
    service = OverviewService(get_settings())
    result = await service.search(topic)
    objects = result.extraction.objects
    return templates.TemplateResponse(
        request,
        "objects.html",
        {
            "title": "Data Objects",
            "topic": topic,
            "run": result,
            "objects": objects,
            "markets": result.extraction.markets,
        },
    )


@router.post("/overview")
async def overview_submit(
    request: Request,
    search_run_id: str = Form(...),
    object_id: str = Form(...),
) -> Response:
    service = OverviewService(get_settings())
    result = service.aggregate_object(search_run_id, object_id)
    return templates.TemplateResponse(
        request,
        "overview.html",
        {"title": "Overview", "result": result, "search_run_id": search_run_id},
    )


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
    try:
        return await service.summarize(search_run_id)
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@router.get("/aggregate/{search_run_id}/{object_id}")
async def aggregate_detail(search_run_id: str, object_id: str) -> AggregationResult:
    service = OverviewService(get_settings())
    try:
        return service.aggregate_object(search_run_id, object_id)
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@router.get("/review/{search_run_id}")
async def review_page(request: Request, search_run_id: str) -> Response:
    repo = get_structured_repo()
    markets = repo.list_effective_for_run(search_run_id)
    return templates.TemplateResponse(
        request,
        "review.html",
        {
            "title": "Review Structured Markets",
            "search_run_id": search_run_id,
            "markets": markets,
            "objects": build_objects(markets),
            "object_types": [item.value for item in ObjectType],
            "operators": [item.value for item in Operator],
        },
    )


@router.post("/review/{search_run_id}")
async def review_save(request: Request, search_run_id: str) -> Response:
    repo = get_structured_repo()
    current_markets = repo.list_effective_for_run(search_run_id)
    form = await request.form()
    reviewed: list[StructuredMarket] = []
    for index, current in enumerate(current_markets):
        threshold_text = str(form.get(f"threshold_value_{index}") or "").strip()
        threshold_value = float(threshold_text) if threshold_text else None
        object_type = ObjectType(str(form.get(f"object_type_{index}") or current.object_type.value))
        operator = Operator(str(form.get(f"operator_{index}") or current.operator.value))
        object_name = str(form.get(f"object_name_{index}") or current.object_name).strip()
        threshold_unit = none_if_blank(form.get(f"threshold_unit_{index}"))
        resolution_date = none_if_blank(form.get(f"resolution_date_{index}"))
        event_title = str(form.get(f"event_title_{index}") or current.event_title).strip()
        object_id = build_object_id(
            object_name,
            object_type,
            event_title,
            resolution_date,
            threshold_unit,
        )
        reviewed.append(
            StructuredMarket.model_validate(
                current.model_dump()
                | {
                    "object_id": object_id,
                    "event_title": event_title,
                    "object_name": object_name,
                    "object_type": object_type,
                    "operator": operator,
                    "threshold_value": threshold_value,
                    "threshold_unit": threshold_unit,
                    "category_value": none_if_blank(form.get(f"category_value_{index}")),
                    "resolution_date": resolution_date,
                    "include": form.get(f"include_{index}") == "on",
                }
            )
        )
    repo.save_review(search_run_id, reviewed)
    return RedirectResponse(f"/review/{search_run_id}", status_code=303)


def get_structured_repo() -> StructuredMarketRepository:
    db = connect(get_settings().db_path)
    migrate(db)
    return StructuredMarketRepository(db)


def none_if_blank(value: object) -> str | None:
    text = str(value or "").strip()
    return text or None
