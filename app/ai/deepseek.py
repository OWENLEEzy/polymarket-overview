import json
from typing import Any, cast

import httpx

from app.ai.prompts import SEARCH_PLANNING_PROMPT, SYNTHESIZE_OVERVIEW_PROMPT
from app.ai.provider import AIProvider
from app.analysis.point_estimate import describe_point_estimates
from app.models import PointEstimate


class DeepSeekProvider(AIProvider):
    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-chat",
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.http_client = http_client or httpx.AsyncClient(timeout=60.0)

    async def plan_search_terms(self, topic: str) -> list[str]:
        payload = await self._chat_json(SEARCH_PLANNING_PROMPT, topic)
        queries = payload.get("queries", [])
        return [str(item) for item in queries if str(item).strip()]

    async def synthesize_overview(self, topic: str, point_estimates: list[PointEstimate]) -> str:
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

    async def _chat_json(self, system_prompt: str, user_content: str) -> dict[str, Any]:
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
                "response_format": {"type": "json_object"},
            },
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        return cast(dict[str, Any], json.loads(content))
