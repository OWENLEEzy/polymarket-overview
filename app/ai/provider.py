from abc import ABC, abstractmethod

from app.models import PointEstimate


class AIProvider(ABC):
    """The LLM does exactly two jobs, both with tiny input and output:
    semantic search-term expansion and the final narrative. Market structuring is
    deterministic (see ``app.analysis.structurer``) and never goes through here."""

    @abstractmethod
    async def plan_search_terms(self, topic: str) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    async def synthesize_overview(self, topic: str, point_estimates: list[PointEstimate]) -> str:
        raise NotImplementedError
