"""Hybrid search: dense similarity and keyword relevance fused by reciprocal rank.

Dense search finds what a query MEANS; keyword search finds what it SAYS (an error code, a name, a part number that an embedding blurs). A store that offers both
implements `HybridSearch.query_hybrid`; this module holds what they share: tokenising a query safely for each backend's text-search syntax, and the fusion.

Fusion is reciprocal rank (`score = sum(1 / (60 + rank))` over the result lists a record appears in), the standard way to combine rankings whose raw scores are not
comparable (a cosine similarity and a BM25 value). The fused `Match.score` is therefore an RRF value: higher is better, and it is not comparable with a dense score."""

from __future__ import annotations

import re
from collections.abc import Sequence

from aaa.vector.types import Match

RRF_K = 60
_WORD = re.compile(r"\w+", re.UNICODE)


def tokens(text: str, *, limit: int = 32) -> list[str]:
    """The distinct word tokens of a query (letters, digits, underscore), in order, safe to put inside any text-search syntax."""
    seen: dict[str, None] = {}
    for w in _WORD.findall(text.lower()):
        seen.setdefault(w, None)
    return list(seen)[:limit]


def fuse(
    dense: Sequence[Match],
    keyword: Sequence[Match],
    k: int,
    *,
    weights: tuple[float, float] = (1.0, 1.0),
) -> list[Match]:
    """Merge two ranked lists into one, best first, keeping each record's payload."""
    score: dict[str, float] = {}
    by_id: dict[str, Match] = {}
    for w, ranked in zip(weights, (dense, keyword), strict=True):
        for rank, m in enumerate(ranked):
            score[m.id] = score.get(m.id, 0.0) + w / (RRF_K + rank + 1)
            by_id.setdefault(m.id, m)
    best = sorted(score, key=lambda i: (-score[i], i))[:k]
    return [Match(i, score[i], by_id[i].text, by_id[i].meta) for i in best]


def pool(k: int) -> int:
    """How many candidates each leg fetches before fusion."""
    return max(k * 4, 50)
