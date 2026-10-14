from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from aaa.data.capabilities import Embedder

_TOKEN = re.compile(r"\w+", re.UNICODE)


class Doc(BaseModel):
    id: str
    text: str
    meta: dict[str, Any] = Field(default_factory=dict)
    embedding: list[float] | None = None


class Hit(BaseModel):
    doc: Doc
    score: float


@runtime_checkable
class Index(Protocol):
    async def upsert(self, docs: list[Doc]) -> None: ...
    async def delete(self, ids: list[str]) -> None: ...
    async def search(
        self, query: str, k: int = 5, *, filter: Callable[[Doc], bool] | None = None
    ) -> list[Hit]: ...
    async def count(self) -> int: ...


def _tok(s: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(s)]


def _cos(a: list[float], b: list[float]) -> float:
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(x * x for x in b))
    return sum(x * y for x, y in zip(a, b, strict=False)) / (na * nb) if na and nb else 0.0


class InMemoryIndex:
    """BM25 keyword search; if an Embedder is supplied, fused with cosine similarity via
    reciprocal-rank fusion (hybrid). Brute force: fine to ~1e5 docs; swap in pgvector/Qdrant
    behind the same protocol for more."""

    def __init__(self, embedder: Embedder | None = None, k1: float = 1.5, b: float = 0.75) -> None:
        self._docs: dict[str, Doc] = {}
        self._tf: dict[str, Counter[str]] = {}
        self._embedder, self.k1, self.b = embedder, k1, b

    async def upsert(self, docs: list[Doc]) -> None:
        missing = [d for d in docs if d.embedding is None]
        if self._embedder and missing:
            vecs = await self._embedder.embed([d.text for d in missing])
            for d, v in zip(missing, vecs, strict=True):
                d.embedding = v
        for d in docs:
            self._docs[d.id] = d
            self._tf[d.id] = Counter(_tok(d.text))

    async def delete(self, ids: list[str]) -> None:
        for i in ids:
            self._docs.pop(i, None)
            self._tf.pop(i, None)

    async def count(self) -> int:
        return len(self._docs)

    def _bm25(self, q: list[str], ids: list[str]) -> dict[str, float]:
        n = len(self._docs)
        avg = sum(sum(c.values()) for c in self._tf.values()) / max(n, 1)
        df = {t: sum(1 for c in self._tf.values() if t in c) for t in set(q)}
        out: dict[str, float] = {}
        for i in ids:
            tf, dl = self._tf[i], sum(self._tf[i].values())
            s = 0.0
            for t in q:
                if tf[t] and df[t]:
                    idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                    s += (
                        idf
                        * tf[t]
                        * (self.k1 + 1)
                        / (tf[t] + self.k1 * (1 - self.b + self.b * dl / avg))
                    )
            if s > 0:
                out[i] = s
        return out

    async def search(
        self, query: str, k: int = 5, *, filter: Callable[[Doc], bool] | None = None
    ) -> list[Hit]:
        ids = [i for i, d in self._docs.items() if filter is None or filter(d)]
        kw = self._bm25(_tok(query), ids)
        ranked = {"kw": sorted(kw, key=kw.__getitem__, reverse=True)}
        if self._embedder:
            qv = (await self._embedder.embed([query]))[0]
            sims = {i: _cos(qv, self._docs[i].embedding or []) for i in ids}
            ranked["vec"] = [
                i for i in sorted(sims, key=sims.__getitem__, reverse=True) if sims[i] > 0
            ]
        if len(ranked) == 1:
            return [Hit(doc=self._docs[i], score=kw[i]) for i in ranked["kw"][:k]]
        fused: dict[str, float] = {}
        for order in ranked.values():
            for rank, i in enumerate(order):
                fused[i] = fused.get(i, 0.0) + 1 / (60 + rank)
        best = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:k]
        return [Hit(doc=self._docs[i], score=s) for i, s in best]
