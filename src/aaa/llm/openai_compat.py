from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from aaa.llm.config import LLMConfig, expand_env
from aaa.llm.errors import (
    AuthError,
    ContentFiltered,
    ContextOverflow,
    FatalError,
    LLMConfigError,
    RateLimited,
    TransientError,
)
from aaa.llm.types import (
    EmbedResponse,
    LLMRequest,
    LLMResponse,
    Message,
    StreamChunk,
    ToolCall,
    ToolCallDelta,
    Usage,
)

_OVERFLOW = ("context_length", "context length", "maximum context", "too many tokens", "too long")


def _msg_out(m: Message) -> dict[str, Any]:
    d: dict[str, Any] = {"role": m.role, "content": m.content}
    if m.tool_calls:
        d["tool_calls"] = [
            {
                "id": t.id,
                "type": "function",
                "function": {"name": t.name, "arguments": json.dumps(t.arguments)},
            }
            for t in m.tool_calls
        ]
    if m.tool_call_id:
        d["tool_call_id"] = m.tool_call_id
    if m.name:
        d["name"] = m.name
    return d


def _parse_tool_call(raw: dict[str, Any]) -> ToolCall:
    fn = raw.get("function", {})
    args = fn.get("arguments") or "{}"
    try:
        parsed = json.loads(args) if isinstance(args, str) else args
        if not isinstance(parsed, dict):
            raise TypeError("arguments not an object")
        return ToolCall(id=raw.get("id", ""), name=fn.get("name", ""), arguments=parsed)
    except (ValueError, TypeError):
        return ToolCall(id=raw.get("id", ""), name=fn.get("name", ""), raw_arguments=str(args))


def classify_http_error(resp: httpx.Response) -> Exception:
    body = resp.text[:500]
    code = resp.status_code
    low = body.lower()
    if code == 429:
        ra = resp.headers.get("retry-after")
        try:
            return RateLimited(body, float(ra) if ra else None)
        except ValueError:
            return RateLimited(body)
    if code in (408, 409, 425) or code >= 500:
        return TransientError(f"HTTP {code}: {body}")
    if code in (401, 403):
        return AuthError(f"HTTP {code}: {body}")
    if "content_filter" in low or "content_policy" in low:
        return ContentFiltered(body)
    if code in (400, 413) and any(s in low for s in _OVERFLOW):
        return ContextOverflow(body)
    return FatalError(f"HTTP {code}: {body}")


class OpenAICompatClient:
    """One adapter for every gateway that speaks OpenAI chat-completions (Bifrost, Portkey,
    Kong, Cloudflare, Vercel, MLflow, LiteLLM proxy, vLLM, Ollama, OpenAI itself ...)."""

    def __init__(self, config: LLMConfig, *, http_client: httpx.AsyncClient | None = None) -> None:
        self.config = config
        self._headers = {k: expand_env(v) for k, v in config.headers.items()}
        if config.api_key:
            self._headers.setdefault("authorization", f"Bearer {expand_env(config.api_key)}")
        self._headers.setdefault("content-type", "application/json")
        self._http = http_client or httpx.AsyncClient(timeout=config.timeout_s)
        self._owns_http = http_client is None
        if not config.base_url:
            raise LLMConfigError("base_url is required for openai_compat")

    def _url(self, path: str) -> str:
        return self.config.base_url.rstrip("/") + path

    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            resp = await self._http.post(self._url(path), json=payload, headers=self._headers)
        except httpx.TimeoutException as e:
            raise TransientError(f"timeout: {e}") from e
        except httpx.TransportError as e:
            raise TransientError(f"transport error: {e}") from e
        if resp.status_code >= 400:
            raise classify_http_error(resp)
        try:
            return resp.json()
        except ValueError as e:
            raise TransientError(f"non-JSON response: {resp.text[:200]}") from e

    def _cost(self, u: Usage) -> float | None:
        c = self.config
        if c.price_in_per_mtok is None and c.price_out_per_mtok is None:
            return None
        price_in = c.price_in_per_mtok or 0
        cached_price = (
            c.price_cached_in_per_mtok if c.price_cached_in_per_mtok is not None else price_in
        )
        fresh = max(0, u.prompt_tokens - u.cached_tokens)
        return (
            fresh * price_in
            + u.cached_tokens * cached_price
            + u.completion_tokens * (c.price_out_per_mtok or 0)
        ) / 1_000_000

    def _payload(self, request: LLMRequest) -> dict[str, Any]:
        model = request.model or self.config.model
        payload: dict[str, Any] = {
            **self.config.params,
            **request.params,
            "model": model,
            "messages": [_msg_out(m) for m in request.messages],
        }
        if request.tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in request.tools
            ]
        if request.response_schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "output",
                    "schema": request.response_schema,
                    "strict": True,
                },
            }
        return payload

    async def complete(self, request: LLMRequest) -> LLMResponse:
        payload = self._payload(request)
        model = payload["model"]
        data = await self._post(self.config.chat_path, payload)
        try:
            choice = data["choices"][0]
            raw = choice["message"]
        except (KeyError, IndexError, TypeError) as e:
            raise TransientError(f"malformed completion response: {str(data)[:200]}") from e
        u = data.get("usage") or {}
        usage = Usage(
            prompt_tokens=u.get("prompt_tokens", 0),
            completion_tokens=u.get("completion_tokens", 0),
            total_tokens=u.get("total_tokens")
            or u.get("prompt_tokens", 0) + u.get("completion_tokens", 0),
            cached_tokens=(u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0,
        )
        if choice.get("finish_reason") == "content_filter":
            raise ContentFiltered("response blocked by content filter")
        msg = Message(
            role="assistant",
            content=raw.get("content"),
            tool_calls=[_parse_tool_call(t) for t in raw.get("tool_calls") or []],
        )
        return LLMResponse(
            message=msg,
            usage=usage,
            cost=self._cost(usage),
            model=data.get("model", model),
            finish_reason=choice.get("finish_reason"),
        )

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:
        """Server-sent-events streaming. Not retried once started (caller can't un-see text)."""
        payload = self._payload(request)
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}
        try:
            async with self._http.stream(
                "POST", self._url(self.config.chat_path), json=payload, headers=self._headers
            ) as resp:
                if resp.status_code >= 400:
                    await resp.aread()
                    raise classify_http_error(resp)
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    body = line[5:].strip()
                    if body == "[DONE]":
                        return
                    try:
                        obj = json.loads(body)
                    except ValueError as e:
                        raise TransientError(f"bad stream frame: {body[:100]}") from e
                    yield self._chunk(obj)
        except httpx.TimeoutException as e:
            raise TransientError(f"timeout: {e}") from e
        except httpx.TransportError as e:
            raise TransientError(f"transport error: {e}") from e

    def _chunk(self, obj: dict[str, Any]) -> StreamChunk:
        choice = (obj.get("choices") or [{}])[0]
        delta = choice.get("delta") or {}
        u = obj.get("usage")
        return StreamChunk(
            text=delta.get("content") or "",
            tool_calls=[
                ToolCallDelta(
                    index=t.get("index", 0),
                    id=t.get("id"),
                    name=(t.get("function") or {}).get("name"),
                    arguments=(t.get("function") or {}).get("arguments") or "",
                )
                for t in delta.get("tool_calls") or []
            ],
            finish_reason=choice.get("finish_reason"),
            usage=Usage(
                prompt_tokens=u.get("prompt_tokens", 0),
                completion_tokens=u.get("completion_tokens", 0),
                total_tokens=u.get("total_tokens", 0),
            )
            if u
            else None,
            model=obj.get("model", ""),
        )

    async def embed(self, texts: list[str], model: str | None = None) -> EmbedResponse:
        m = model or self.config.model
        data = await self._post(self.config.embeddings_path, {"model": m, "input": texts})
        try:
            vectors = [
                d["embedding"] for d in sorted(data["data"], key=lambda d: d.get("index", 0))
            ]
        except (KeyError, TypeError) as e:
            raise TransientError("malformed embeddings response") from e
        u = data.get("usage") or {}
        return EmbedResponse(
            vectors=vectors,
            model=data.get("model", m),
            usage=Usage(
                prompt_tokens=u.get("prompt_tokens", 0), total_tokens=u.get("total_tokens", 0)
            ),
        )

    async def aclose(self) -> None:
        if self._owns_http:
            await self._http.aclose()
