"""An in-process reference store: brute force, exact. The oracle the contract tests compare real stores against."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

from aaa.vector import hybrid
from aaa.vector.filters import Filter, matches
from aaa.vector.types import DEFAULT_NAMESPACE, Match, Metric, Record, check_vector


def score_of(metric: Metric, a: Sequence[float], b: Sequence[float]) -> float:
    if metric == "euclidean":
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b, strict=True)))
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    if metric == "dot":
        return dot
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class InMemoryVectorStore:
    def __init__(self, dimension: int | None = None, metric: Metric = "cosine") -> None:
        self.dimension: int | None = dimension
        self.metric: Metric = metric
        self._data: dict[str, dict[str, Record]] = {}

    @property
    def higher_is_better(self) -> bool:
        return self.metric != "euclidean"

    def _ns(self, namespace: str | None) -> dict[str, Record]:
        return self._data.setdefault(namespace or DEFAULT_NAMESPACE, {})

    async def upsert(self, records: Sequence[Record], *, namespace: str | None = None) -> None:
        for r in records:
            self.dimension = check_vector(self.dimension, r.vector)
        ns = self._ns(namespace)
        for r in records:
            ns[r.id] = Record(r.id, list(r.vector), r.text, dict(r.meta))

    async def query(
        self,
        vector: Sequence[float],
        k: int = 5,
        *,
        filter: Filter | None = None,
        namespace: str | None = None,
    ) -> list[Match]:
        check_vector(self.dimension, vector, "query vector")
        scored = [
            (score_of(self.metric, vector, r.vector), r)
            for r in self._ns(namespace).values()
            if matches(filter, r.meta)
        ]
        scored.sort(key=lambda t: t[0], reverse=self.higher_is_better)
        return [Match(r.id, s, r.text, dict(r.meta)) for s, r in scored[:k]]

    async def query_hybrid(
        self,
        vector: Sequence[float],
        text: str,
        k: int = 5,
        *,
        filter: Filter | None = None,
        namespace: str | None = None,
    ) -> list[Match]:
        """Dense + BM25 (the reference the other stores are compared with)."""
        dense = await self.query(vector, hybrid.pool(k), filter=filter, namespace=namespace)
        words = hybrid.tokens(text)
        docs = [
            (r, Counter(hybrid.tokens(r.text, limit=10**6)))
            for r in self._ns(namespace).values()
            if matches(filter, r.meta)
        ]
        n = len(docs)
        avg = sum(sum(c.values()) for _, c in docs) / max(n, 1)
        df = {w: sum(1 for _, c in docs if w in c) for w in words}
        scored = []
        for r, c in docs:
            dl, s = sum(c.values()), 0.0
            for w in words:
                if c[w] and df[w]:
                    idf = math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5))
                    s += idf * c[w] * 2.5 / (c[w] + 1.5 * (0.25 + 0.75 * dl / max(avg, 1e-9)))
            if s > 0:
                scored.append(Match(r.id, s, r.text, dict(r.meta)))
        scored.sort(key=lambda m: (-m.score, m.id))
        return hybrid.fuse(dense, scored[: hybrid.pool(k)], k)

    async def get(self, ids: Sequence[str], *, namespace: str | None = None) -> list[Record]:
        ns = self._ns(namespace)
        return [Record(r.id, list(r.vector), r.text, dict(r.meta)) for i in ids if (r := ns.get(i))]

    async def delete(self, ids: Sequence[str], *, namespace: str | None = None) -> None:
        ns = self._ns(namespace)
        for i in ids:
            ns.pop(i, None)

    async def delete_where(self, filter: Filter, *, namespace: str | None = None) -> int:
        ns = self._ns(namespace)
        doomed = [i for i, r in ns.items() if matches(filter, r.meta)]
        for i in doomed:
            del ns[i]
        return len(doomed)

    async def drop_namespace(self, namespace: str) -> None:
        self._data.pop(namespace, None)

    async def count(self, *, namespace: str | None = None) -> int:
        return len(self._ns(namespace))

    async def aclose(self) -> None:
        return None
