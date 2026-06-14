import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]

from app.models import RawMarket


TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


@dataclass(frozen=True)
class RankedMarket:
    market: RawMarket
    score: float


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_RE.finditer(text)]


def market_text(market: RawMarket) -> str:
    return " ".join(
        part
        for part in [market.question, market.event_title or "", " ".join(market.outcomes)]
        if part
    )


def rank_markets(query: str, markets: list[RawMarket]) -> list[RankedMarket]:
    if not markets:
        return []
    corpus = [tokenize(market_text(market)) for market in markets]
    bm25 = BM25Okapi(corpus)
    query_tokens = tokenize(query)
    query_token_set = set(query_tokens)
    scores = bm25.get_scores(tokenize(query))
    ranked = [
        RankedMarket(
            market=market,
            score=float(score) + len(query_token_set.intersection(set(corpus_tokens))),
        )
        for market, score, corpus_tokens in zip(markets, scores, corpus, strict=True)
    ]
    return sorted(ranked, key=lambda item: item.score, reverse=True)
