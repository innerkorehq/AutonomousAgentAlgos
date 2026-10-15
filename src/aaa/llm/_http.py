"""Plumbing shared by the native provider clients (Anthropic, Gemini): an httpx client, error mapping hooks, cost, and server-sent events."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx

from aaa.llm.config import LLMConfig, expand_env
from aaa.llm.errors import LLMConfigError, TransientError
from aaa.llm.types import Usage

Classify = Callable[[httpx.Response], Exception]


class HTTPClientBase:
    modalities: frozenset[str]

    def __init__(
        self,
        config: LLMConfig,
        *,
        http_client: httpx.AsyncClient | None,
        base_url: str,
        auth_header: str,
        default_modalities: tuple[str, ...],
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.config = config
        self.base_url = (config.base_url or base_url).rstrip("/")
        if not self.base_url:
            raise LLMConfigError("base_url is required")
        self.headers = {k: expand_env(v) for k, v in config.headers.items()}
        if config.api_key:
            self.headers.setdefault(auth_header, expand_env(config.api_key))
        for k, v in (extra_headers or {}).items():
            self.headers.setdefault(k, v)
        self.headers.setdefault("content-type", "application/json")
        self.modalities = frozenset(config.modalities or default_modalities)
        self._http = http_client or httpx.AsyncClient(timeout=config.timeout_s)
        self._owns_http = http_client is None

    def cost(self, u: Usage) -> float | None:
        c = self.config
        if c.price_in_per_mtok is None and c.price_out_per_mtok is None:
            return None
        price_in = c.price_in_per_mtok or 0
        cached = c.price_cached_in_per_mtok if c.price_cached_in_per_mtok is not None else price_in
        fresh = max(0, u.prompt_tokens - u.cached_tokens)
        return (
            fresh * price_in
            + u.cached_tokens * cached
            + u.completion_tokens * (c.price_out_per_mtok or 0)
        ) / 1_000_000

    async def post_json(
        self, path: str, payload: dict[str, Any], classify: Classify
    ) -> dict[str, Any]:
        try:
            resp = await self._http.post(self.base_url + path, json=payload, headers=self.headers)
        except httpx.TimeoutException as e:
            raise TransientError(f"timeout: {e}") from e
        except httpx.TransportError as e:
            raise TransientError(f"transport error: {e}") from e
        if resp.status_code >= 400:
            raise classify(resp)
        try:
            return resp.json()
        except ValueError as e:
            raise TransientError(f"non-JSON response: {resp.text[:200]}") from e

    async def sse(
        self, path: str, payload: dict[str, Any], classify: Classify
    ) -> AsyncIterator[tuple[str, dict[str, Any]]]:
        """Yield `(event name, decoded data)` for each server-sent event. Not retried once started (the caller cannot un-see text)."""
        try:
            async with self._http.stream(
                "POST", self.base_url + path, json=payload, headers=self.headers
            ) as resp:
                if resp.status_code >= 400:
                    await resp.aread()
                    raise classify(resp)
                event, data = "", []
                async for line in resp.aiter_lines():
                    if line == "":
                        if data:
                            body = "\n".join(data)
                            try:
                                yield event or "message", json.loads(body)
                            except ValueError as e:
                                raise TransientError(f"bad stream frame: {body[:100]}") from e
                        event, data = "", []
                    elif line.startswith("event:"):
                        event = line[6:].strip()
                    elif line.startswith("data:"):
                        data.append(line[5:].lstrip())
                if data:  # a stream that ends without the blank line
                    yield event or "message", json.loads("\n".join(data))
        except httpx.TimeoutException as e:
            raise TransientError(f"timeout: {e}") from e
        except httpx.TransportError as e:
            raise TransientError(f"transport error: {e}") from e

    async def aclose(self) -> None:
        if self._owns_http:
            await self._http.aclose()
