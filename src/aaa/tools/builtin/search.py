"""Web search behind a small protocol. The query leaves your process (to a SearXNG you run, or to a hosted API); the provider's key never reaches the model."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol, runtime_checkable
from urllib.parse import urlsplit

import httpx

from aaa.tools.tool import ToolError

SNIPPET_CHARS = 500


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str


@runtime_checkable
class SearchProvider(Protocol):
    async def search(self, query: str, k: int = 5) -> list[SearchResult]: ...


def clean(results: list[SearchResult], k: int) -> list[SearchResult]:
    """Only http(s) URLs, snippets capped: results are untrusted text."""
    out = []
    for r in results:
        if urlsplit(r.url).scheme.lower() in ("http", "https") and r.url:
            out.append(
                SearchResult(r.title.strip()[:200], r.url, r.snippet.strip()[:SNIPPET_CHARS])
            )
        if len(out) >= k:
            break
    return out


class SearXNGSearch:
    """A SearXNG instance you run (`GET /search?q=...&format=json`). The instance must have `json` in `search.formats` of its settings.yml, otherwise it answers 403."""

    def __init__(
        self,
        base_url: str,
        *,
        categories: str | None = None,
        language: str | None = None,
        safesearch: int | None = None,
        timeout_s: float = 10.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.params = {
            k: v
            for k, v in {
                "categories": categories,
                "language": language,
                "safesearch": safesearch,
            }.items()
            if v is not None
        }
        self._http = httpx.AsyncClient(timeout=timeout_s, transport=transport, trust_env=False)

    async def search(self, query: str, k: int = 5) -> list[SearchResult]:
        try:
            r = await self._http.get(
                f"{self.base_url}/search", params={"q": query, "format": "json", **self.params}
            )
        except httpx.HTTPError as e:
            raise ToolError(f"search failed: {type(e).__name__}") from e
        if r.status_code == 403:
            raise ToolError(
                "the SearXNG instance does not allow format=json (add it to search.formats in settings.yml)"
            )
        if r.status_code >= 400:
            raise ToolError(f"search failed: HTTP {r.status_code}")
        try:
            items = r.json().get("results") or []
        except ValueError as e:
            raise ToolError("search returned a response that is not JSON") from e
        return clean(
            [
                SearchResult(
                    str(i.get("title") or ""), str(i.get("url") or ""), str(i.get("content") or "")
                )
                for i in items
            ],
            k,
        )

    async def aclose(self) -> None:
        await self._http.aclose()


class BraveSearch:
    """Brave Search API (`GET https://api.search.brave.com/res/v1/web/search`, header `X-Subscription-Token`). Built from Brave's published request/response description
    and tested with a mock transport; not run against a live account."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str = "https://api.search.brave.com",
        safesearch: str = "moderate",
        country: str | None = None,
        timeout_s: float = 10.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("BRAVE_API_KEY")
        self.base_url, self.safesearch, self.country = base_url.rstrip("/"), safesearch, country
        self._http = httpx.AsyncClient(timeout=timeout_s, transport=transport, trust_env=False)

    async def search(self, query: str, k: int = 5) -> list[SearchResult]:
        params: dict[str, str | int] = {
            "q": query,
            "count": min(max(k, 1), 20),
            "safesearch": self.safesearch,
        }
        if self.country:
            params["country"] = self.country
        try:
            r = await self._http.get(
                f"{self.base_url}/res/v1/web/search",
                params=params,
                headers={"X-Subscription-Token": self.api_key or "", "Accept": "application/json"},
            )
        except httpx.HTTPError as e:
            raise ToolError(f"search failed: {type(e).__name__}") from e
        if r.status_code in (401, 403):
            raise ToolError("the search provider rejected the API key")
        if r.status_code == 429:
            raise ToolError("the search provider's rate limit was reached")
        if r.status_code >= 400:
            raise ToolError(f"search failed: HTTP {r.status_code}")
        items = ((r.json().get("web") or {}).get("results")) or []
        return clean(
            [
                SearchResult(
                    str(i.get("title") or ""),
                    str(i.get("url") or ""),
                    str(i.get("description") or ""),
                )
                for i in items
            ],
            k,
        )

    async def aclose(self) -> None:
        await self._http.aclose()
