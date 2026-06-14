import argparse
import asyncio
import json

from app.config import get_settings
from app.service import OverviewService


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("topic")
    parser.add_argument("--object", dest="object_name")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()

    service = OverviewService(get_settings())
    result = await service.search(args.topic, limit=args.limit)
    if args.summary:
        print(
            json.dumps(
                {
                    "search_run_id": result.search_run_id,
                    "topic": result.topic,
                    "search_terms": result.search_terms,
                    "raw_market_count": len(result.raw_markets),
                    "objects": [item.model_dump() for item in result.extraction.objects],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
    else:
        print(
            json.dumps(
                {
                    "search_run_id": result.search_run_id,
                    "topic": result.topic,
                    "search_terms": result.search_terms,
                    "raw_market_count": len(result.raw_markets),
                    "objects": [item.model_dump() for item in result.extraction.objects],
                    "markets": [item.model_dump() for item in result.extraction.markets],
                },
                indent=2,
                ensure_ascii=False,
            )
        )

    object_name = args.object_name
    if object_name is None and not args.summary and result.extraction.objects:
        object_name = result.extraction.objects[0].object_name
    if object_name:
        aggregation = service.aggregate_object(result.search_run_id, object_name)
        print(json.dumps({"aggregation": aggregation.model_dump()}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
