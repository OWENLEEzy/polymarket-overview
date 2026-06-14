from app.ai.deepseek import DeepSeekProvider
from app.ai.fallback import RuleBasedProvider
from app.ai.provider import AIProvider
from app.analysis.aggregators import aggregate
from app.analysis.point_estimate import compute_point_estimate
from app.analysis.structurer import structure_markets
from app.config import Settings
from app.models import (
    AggregationInput,
    AggregationResult,
    ObjectType,
    OverviewSummary,
    PointEstimate,
    SearchResult,
    StructuredMarket,
)
from app.polymarket.client import PolymarketClient
from app.recall.ranker import rank_markets
from app.storage.db import connect, migrate
from app.storage.repositories import (
    AggregationRunRepository,
    RawMarketRepository,
    SearchRunRepository,
    StructuredMarketRepository,
)


def build_ai_provider(settings: Settings) -> AIProvider:
    if settings.ai_provider == "deepseek" and settings.deepseek_api_key:
        return DeepSeekProvider(settings.deepseek_api_key, model=settings.deepseek_model)
    return RuleBasedProvider()


class OverviewService:
    def __init__(
        self,
        settings: Settings,
        polymarket_client: PolymarketClient | None = None,
        ai_provider: AIProvider | None = None,
    ) -> None:
        self.settings = settings
        self.polymarket_client = polymarket_client or PolymarketClient()
        self.ai_provider = ai_provider or build_ai_provider(settings)

    async def search(self, topic: str, limit: int = 20) -> SearchResult:
        search_terms = await self.ai_provider.plan_search_terms(topic)
        if topic not in search_terms:
            search_terms.insert(0, topic)
        raw_markets = await self.polymarket_client.search_many(search_terms)
        ranked = rank_markets(topic, raw_markets)
        selected = [item.market for item in ranked[:limit]]
        extraction = structure_markets(topic, selected)

        db = connect(self.settings.db_path)
        migrate(db)
        provider_name = type(self.ai_provider).__name__
        search_run_id = SearchRunRepository(db).create(topic, provider_name, search_terms)
        RawMarketRepository(db).save_many(search_run_id, selected)
        StructuredMarketRepository(db).save_many(search_run_id, extraction.markets)

        return SearchResult(
            search_run_id=search_run_id,
            topic=topic,
            search_terms=search_terms,
            raw_markets=selected,
            extraction=extraction,
        )

    def aggregate_object(self, search_run_id: str, object_id: str) -> AggregationResult:
        db = connect(self.settings.db_path)
        migrate(db)
        all_markets = StructuredMarketRepository(db).list_effective_for_run(search_run_id)
        markets = [
            market
            for market in all_markets
            if (market.object_id == object_id or market.object_name == object_id) and market.include
        ]
        if not markets:
            raise ValueError(f"No included markets for object {object_id}")
        object_type = markets[0].object_type
        if any(market.object_type != object_type for market in markets):
            object_type = ObjectType.CATEGORICAL
        payload = AggregationInput(
            object_id=markets[0].object_id,
            object_name=markets[0].object_name,
            object_type=object_type,
            markets=markets,
        )
        result = aggregate(payload)
        result.anomalies.extend(definition_anomalies(markets))
        AggregationRunRepository(db).create(search_run_id, payload, result)
        return result

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


def _ordered_object_ids(markets: list[StructuredMarket]) -> list[str]:
    seen: dict[str, None] = {}
    for market in markets:
        seen.setdefault(market.object_id or market.object_name, None)
    return list(seen)


def definition_anomalies(markets: list[StructuredMarket]) -> list[str]:
    anomalies: list[str] = []
    if len({market.object_type for market in markets}) > 1:
        anomalies.append("mixed_object_types")
    if len({market.event_title for market in markets}) > 1:
        anomalies.append("mixed_event_titles")
    if len({(market.resolution_date or "").split("T")[0] for market in markets}) > 1:
        anomalies.append("mixed_resolution_dates")
    threshold_units = {
        market.threshold_unit for market in markets if market.threshold_value is not None
    }
    if len(threshold_units) > 1:
        anomalies.append("mixed_threshold_units")
    return anomalies
