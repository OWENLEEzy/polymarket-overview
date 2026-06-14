from app.ai.provider import AIProvider
from app.analysis.point_estimate import ordered_by_role
from app.models import PointEstimate


class RuleBasedProvider(AIProvider):
    """No-API-key fallback. Search-term expansion is a few heuristics; the narrative
    is templated from the already-computed point estimates."""

    async def plan_search_terms(self, topic: str) -> list[str]:
        base = topic.strip()
        terms = [base]
        lowered = base.lower()
        if "ipo" in lowered:
            terms.extend([f"{base} valuation", f"{base} public", f"{base} market cap"])
        return list(dict.fromkeys(terms))

    async def synthesize_overview(self, topic: str, point_estimates: list[PointEstimate]) -> str:
        ordered = ordered_by_role(point_estimates)
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
