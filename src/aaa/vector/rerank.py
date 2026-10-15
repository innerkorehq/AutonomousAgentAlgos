"""Second-stage ranking: fetch generously, then reorder with a better (slower) judge."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any, Protocol

from aaa.data.index import Doc, Hit, Index
from aaa.llm.client import LLMClient
from aaa.llm.types import LLMRequest, Message
from aaa.vector.filters import Filter


class Reranker(Protocol):
    async def rerank(self, query: str, hits: list[Hit], k: int) -> list[Hit]: ...


class LLMReranker:
    """Listwise rerank with any `LLMClient`: the model sees the query and numbered passages and returns the best order.
    Unreadable replies keep the original order (reranking is an improvement, never a failure)."""

    def __init__(
        self, client: LLMClient, *, model: str | None = None, max_chars: int = 800
    ) -> None:
        self.client, self.model, self.max_chars = client, model, max_chars

    async def rerank(self, query: str, hits: list[Hit], k: int) -> list[Hit]:
        if len(hits) <= 1:
            return hits[:k]
        listing = "\n".join(f"[{i}] {h.doc.text[: self.max_chars]}" for i, h in enumerate(hits))
        resp = await self.client.complete(
            LLMRequest(
                model=self.model,
                messages=[
                    Message(
                        role="system",
                        content='Rank the passages by how well they answer the query. Reply with JSON only: {"order": [<passage numbers, best first>]}.',
                    ),
                    Message(role="user", content=f"Query: {query}\n\nPassages:\n{listing}"),
                ],
                response_schema={
                    "type": "object",
                    "properties": {"order": {"type": "array", "items": {"type": "integer"}}},
                    "required": ["order"],
                },
                params={"temperature": 0},
            )
        )
        try:
            m = re.search(r"\{.*\}", resp.message.content or "", re.DOTALL)
            order = [
                i
                for i in json.loads(m.group() if m else "")["order"]
                if isinstance(i, int) and 0 <= i < len(hits)
            ]
        except (ValueError, KeyError, TypeError):
            return hits[:k]
        seen: list[int] = []
        for i in order + list(range(len(hits))):
            if i not in seen:
                seen.append(i)
        return [hits[i] for i in seen[:k]]


class RerankingIndex:
    """`RerankingIndex(index, reranker, fetch_k=30)`: retrieve `fetch_k`, return the reranker's best `k`. Scores of the returned hits
    are the first-stage scores (the reranker's own are not comparable to them)."""

    def __init__(self, index: Index, reranker: Reranker, *, fetch_k: int = 30) -> None:
        self.index, self.reranker, self.fetch_k = index, reranker, fetch_k

    async def upsert(self, docs: list[Doc]) -> None:
        await self.index.upsert(docs)

    async def delete(self, ids: list[str]) -> None:
        await self.index.delete(ids)

    async def count(self) -> int:
        return await self.index.count()

    async def search(
        self,
        query: str,
        k: int = 5,
        *,
        filter: Callable[[Doc], bool] | Filter | dict[str, Any] | None = None,
    ) -> list[Hit]:
        hits = await self.index.search(query, max(self.fetch_k, k), filter=filter)  # type: ignore[call-arg]
        return await self.reranker.rerank(query, hits, k)


class CohereReranker:
    """Cohere's rerank endpoint (`POST https://api.cohere.com/v2/rerank`; plain HTTP, no SDK). The passages leave your process.

        RerankingIndex(index, CohereReranker(api_key="...", model="rerank-v4.0-pro"), fetch_k=30)

    Built from Cohere's published request/response description and tested with a mock transport; not run against a live account."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        model: str = "rerank-v4.0-pro",
        base_url: str = "https://api.cohere.com",
        timeout_s: float = 20.0,
        transport: Any = None,
    ) -> None:
        import os

        import httpx

        self.api_key = api_key or os.environ.get("COHERE_API_KEY")
        self.model, self.base_url = model, base_url.rstrip("/")
        self._http = httpx.AsyncClient(timeout=timeout_s, transport=transport)

    async def rerank(self, query: str, hits: list[Hit], k: int) -> list[Hit]:
        if len(hits) <= 1:
            return hits[:k]
        r = await self._http.post(
            f"{self.base_url}/v2/rerank",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "query": query,
                "documents": [h.doc.text for h in hits],
                "top_n": k,
            },
        )
        r.raise_for_status()
        order = [x["index"] for x in r.json()["results"] if 0 <= x["index"] < len(hits)]
        return [hits[i] for i in order[:k]]

    async def aclose(self) -> None:
        await self._http.aclose()


class VoyageReranker:
    """Voyage AI's rerank endpoint (`POST https://api.voyageai.com/v1/rerank`; plain HTTP, no SDK). The passages leave your process.

        RerankingIndex(index, VoyageReranker(api_key="...", model="rerank-2.5"), fetch_k=30)

    Built from Voyage's published request/response description (`data[].index`, `data[].relevance_score`) and tested with a mock transport; not run against a live account.
    (A Jina reranker is not included: Jina's documentation does not specify the response format.)"""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        model: str = "rerank-2.5",
        base_url: str = "https://api.voyageai.com",
        timeout_s: float = 20.0,
        transport: Any = None,
    ) -> None:
        import os

        import httpx

        self.api_key = api_key or os.environ.get("VOYAGE_API_KEY")
        self.model, self.base_url = model, base_url.rstrip("/")
        self._http = httpx.AsyncClient(timeout=timeout_s, transport=transport)

    async def rerank(self, query: str, hits: list[Hit], k: int) -> list[Hit]:
        if len(hits) <= 1:
            return hits[:k]
        r = await self._http.post(
            f"{self.base_url}/v1/rerank",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "query": query,
                "documents": [h.doc.text for h in hits],
                "top_k": k,
            },
        )
        r.raise_for_status()
        order = [x["index"] for x in r.json()["data"] if 0 <= x["index"] < len(hits)]
        return [hits[i] for i in order[:k]]

    async def aclose(self) -> None:
        await self._http.aclose()
