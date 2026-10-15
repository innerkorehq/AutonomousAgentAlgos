"""Native Anthropic Messages API client (`provider="anthropic"`), over httpx: no vendor SDK needed.

    LLMConfig(provider="anthropic", model="claude-sonnet-5-5", api_key="${ANTHROPIC_API_KEY}", params={"max_tokens": 2048})

What it maps:

* system messages -> the separate `system` field; consecutive tool results -> one user turn of `tool_result` blocks (before any user text);
* tool calls <-> `tool_use` blocks, with the call ids kept so results pair up;
* `response_schema` -> a forced tool (the schema is the tool's input schema); its input is returned as the reply's JSON text;
* attachments: images (base64 or https URL) and PDFs (`document` blocks; plain text files as text documents). Audio is not supported by the Messages API: a request
  with audio fails with `UnsupportedModality`;
* usage: `prompt_tokens` includes cache reads and writes, `cached_tokens` is the cache-read part; cost from the `price_*` config;
* `cache=True` puts `cache_control` breakpoints on the system prompt, the tools and the latest user content (Anthropic prompt caching);
* streaming (`stream`), embeddings are not offered (Anthropic has no embeddings API).

`max_tokens` is REQUIRED by the API: take it from `params`, else `default_max_tokens`. Parameters Anthropic does not know are dropped with a warning log
rather than sent (the API rejects unknown fields). Extended-thinking blocks are neither requested nor round-tripped."""

from __future__ import annotations

import base64
import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx

from aaa.llm._http import HTTPClientBase
from aaa.llm.config import LLMConfig
from aaa.llm.errors import (
    AuthError,
    ContentFiltered,
    ContextOverflow,
    FatalError,
    LLMConfigError,
    RateLimited,
    TransientError,
)
from aaa.llm.openai_compat import classify_http_error
from aaa.llm.parts import MediaPart
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

log = logging.getLogger("aaa.llm")

API_VERSION = "2023-06-01"
RESPOND_TOOL = "aaa_respond"
_PARAMS = {
    "temperature",
    "top_p",
    "top_k",
    "stop_sequences",
    "metadata",
    "tool_choice",
    "service_tier",
    "thinking",
    "max_tokens",
}
_STOP = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
    "refusal": "content_filter",
    "pause_turn": "pause_turn",
}


def classify_error(resp: httpx.Response) -> Exception:
    """Anthropic errors are `{"type": "error", "error": {"type", "message"}}`; map them onto AAA's error classes."""
    try:
        err = resp.json().get("error") or {}
    except ValueError:
        err = {}
    kind, msg, code = err.get("type", ""), err.get("message") or resp.text[:300], resp.status_code
    if code == 429 or kind == "rate_limit_error":
        ra = resp.headers.get("retry-after")
        try:
            return RateLimited(msg, float(ra) if ra else None)
        except ValueError:
            return RateLimited(msg)
    if kind in ("overloaded_error", "api_error", "timeout_error") or code in (
        408,
        409,
        500,
        502,
        503,
        504,
        529,
    ):
        return TransientError(f"HTTP {code} {kind}: {msg}")
    if kind in ("authentication_error", "permission_error", "billing_error") or code in (
        401,
        402,
        403,
    ):
        return AuthError(f"HTTP {code} {kind}: {msg}")
    if code == 413 or (
        code == 400
        and any(
            s in msg.lower()
            for s in (
                "prompt is too long",
                "too many tokens",
                "exceed the context",
                "context window",
            )
        )
    ):
        return ContextOverflow(msg)
    if "content_filter" in msg.lower() or kind == "refusal":
        return ContentFiltered(msg)
    return (
        classify_http_error(resp)
        if code >= 400 and not kind
        else FatalError(f"HTTP {code} {kind}: {msg}")
    )


def _media_block(p: MediaPart) -> dict[str, Any]:
    if p.type == "image":
        src = (
            {"type": "url", "url": p.url}
            if p.url
            else {"type": "base64", "media_type": p.media_type, "data": p.data}
        )
        return {"type": "image", "source": src}
    if p.type == "file":
        if p.media_type == "application/pdf":
            src = (
                {"type": "url", "url": p.url}
                if p.url
                else {"type": "base64", "media_type": p.media_type, "data": p.data}
            )
            block: dict[str, Any] = {"type": "document", "source": src}
        elif p.media_type == "text/plain" and p.data is not None:
            block = {
                "type": "document",
                "source": {
                    "type": "text",
                    "media_type": "text/plain",
                    "data": base64.b64decode(p.data).decode("utf-8", "replace"),
                },
            }
        else:
            raise FatalError(f"Anthropic accepts PDF and plain-text documents, not {p.media_type}")
        if p.name:
            block["title"] = p.name
        return block
    raise FatalError(f"Anthropic does not accept {p.type} attachments")


def to_anthropic(
    messages: list[Message], *, cache: bool = False
) -> tuple[list[dict[str, Any]] | None, list[dict[str, Any]]]:
    """(`system` blocks, `messages`) for the Messages API."""
    system: list[str] = []
    turns: list[dict[str, Any]] = []

    def add(role: str, blocks: list[dict[str, Any]]) -> None:
        if not blocks:
            return
        if turns and turns[-1]["role"] == role:
            turns[-1]["content"].extend(blocks)  # the API wants alternating turns: merge neighbours
        else:
            turns.append({"role": role, "content": list(blocks)})

    for m in messages:
        if m.role == "system":
            if m.content:
                system.append(m.content)
        elif m.role == "tool":
            add(
                "user",
                [
                    {
                        "type": "tool_result",
                        "tool_use_id": m.tool_call_id or "",
                        "content": m.content or "",
                    }
                ],
            )
        elif m.role == "assistant":
            blocks: list[dict[str, Any]] = (
                [{"type": "text", "text": m.content}] if m.content else []
            )
            blocks += [
                {"type": "tool_use", "id": t.id, "name": t.name, "input": t.arguments}
                for t in m.tool_calls
            ]
            add("assistant", blocks)
        else:
            blocks = [{"type": "text", "text": m.content}] if m.content else []
            blocks += [_media_block(p) for p in m.parts or []]
            add("user", blocks)
    for t in turns:  # tool results must come first within a user turn
        if t["role"] == "user":
            t["content"].sort(key=lambda b: b["type"] != "tool_result")
    if cache and turns:
        for t in reversed(turns):
            if t["role"] == "user":
                t["content"][-1]["cache_control"] = {"type": "ephemeral"}
                break
    sys_blocks: list[dict[str, Any]] | None = (
        [{"type": "text", "text": "\n\n".join(system)}] if system else None
    )
    if cache and sys_blocks:
        sys_blocks[-1]["cache_control"] = {"type": "ephemeral"}
    return sys_blocks, turns


class AnthropicClient(HTTPClientBase):
    def __init__(
        self,
        config: LLMConfig,
        *,
        http_client: httpx.AsyncClient | None = None,
        path: str = "/v1/messages",
        default_max_tokens: int = 4096,
        cache: bool = False,
        anthropic_version: str = API_VERSION,
        beta: str | None = None,
    ) -> None:
        extra = {
            "anthropic-version": anthropic_version,
            **({"anthropic-beta": beta} if beta else {}),
        }
        super().__init__(
            config,
            http_client=http_client,
            base_url="https://api.anthropic.com",
            auth_header="x-api-key",
            default_modalities=("text", "image", "file"),
            extra_headers=extra,
        )
        self.path, self.default_max_tokens, self.cache = path, default_max_tokens, cache

    # -- request ----------------------------------------------------------------------------
    def _payload(self, request: LLMRequest) -> dict[str, Any]:
        system, turns = to_anthropic(request.messages, cache=self.cache)
        params = {**self.config.params, **request.params}
        if "stop" in params:
            params["stop_sequences"] = (
                [params.pop("stop")] if isinstance(params["stop"], str) else params.pop("stop")
            )
        unknown = sorted(set(params) - _PARAMS)
        if unknown:
            log.warning(
                "anthropic: dropping parameters the Messages API does not take: %s", unknown
            )
        payload: dict[str, Any] = {k: v for k, v in params.items() if k in _PARAMS}
        payload.setdefault("max_tokens", self.default_max_tokens)
        payload.update(model=request.model or self.config.model, messages=turns)
        if system:
            payload["system"] = system
        tools = [
            {"name": t.name, "description": t.description, "input_schema": t.parameters}
            for t in request.tools
        ]
        if request.response_schema:
            tools.append(
                {
                    "name": RESPOND_TOOL,
                    "description": "Give your final answer in exactly this format.",
                    "input_schema": request.response_schema,
                }
            )
            payload["tool_choice"] = (
                {"type": "any"} if request.tools else {"type": "tool", "name": RESPOND_TOOL}
            )
        if tools:
            if self.cache:
                tools[-1] = {**tools[-1], "cache_control": {"type": "ephemeral"}}
            payload["tools"] = tools
        return payload

    # -- response ---------------------------------------------------------------------------
    def _usage(self, u: dict[str, Any]) -> Usage:
        cache_read = u.get("cache_read_input_tokens") or 0
        prompt = (
            (u.get("input_tokens") or 0) + cache_read + (u.get("cache_creation_input_tokens") or 0)
        )
        out = u.get("output_tokens") or 0
        return Usage(
            prompt_tokens=prompt,
            completion_tokens=out,
            total_tokens=prompt + out,
            cached_tokens=cache_read,
        )

    def _parse(self, data: dict[str, Any], model: str) -> LLMResponse:
        if not isinstance(data.get("content"), list):
            raise TransientError(f"malformed message response: {str(data)[:200]}")
        texts: list[str] = []
        calls: list[ToolCall] = []
        structured: str | None = None
        for b in data["content"]:
            if b.get("type") == "text":
                texts.append(b.get("text", ""))
            elif b.get("type") == "tool_use":
                if b.get("name") == RESPOND_TOOL:
                    structured = json.dumps(b.get("input") or {})
                else:
                    calls.append(
                        ToolCall(
                            id=b.get("id", ""),
                            name=b.get("name", ""),
                            arguments=b.get("input") or {},
                        )
                    )
        stop = data.get("stop_reason")
        if stop == "refusal":
            raise ContentFiltered("the model declined to answer")
        content = structured if structured is not None else ("".join(texts) or None)
        usage = self._usage(data.get("usage") or {})
        finish = "stop" if structured is not None and not calls else _STOP.get(stop or "", stop)
        return LLMResponse(
            message=Message(role="assistant", content=content, tool_calls=calls),
            usage=usage,
            cost=self.cost(usage),
            model=data.get("model", model),
            finish_reason=finish,
        )

    async def complete(self, request: LLMRequest) -> LLMResponse:
        payload = self._payload(request)
        return self._parse(
            await self.post_json(self.path, payload, classify_error), payload["model"]
        )

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:
        payload = self._payload(request)
        payload["stream"] = True
        index_of: dict[int, int] = {}  # content-block index -> tool-call index
        model, prompt = payload["model"], 0
        async for event, d in self.sse(self.path, payload, classify_error):
            if event == "message_start":
                msg = d.get("message") or {}
                model = msg.get("model", model)
                u = self._usage(msg.get("usage") or {})
                prompt = u.prompt_tokens
                yield StreamChunk(model=model)
            elif event == "content_block_start":
                b = d.get("content_block") or {}
                if b.get("type") == "tool_use":
                    index_of[d["index"]] = len(index_of)
                    yield StreamChunk(
                        tool_calls=[
                            ToolCallDelta(
                                index=index_of[d["index"]], id=b.get("id"), name=b.get("name")
                            )
                        ]
                    )
            elif event == "content_block_delta":
                delta = d.get("delta") or {}
                if delta.get("type") == "text_delta":
                    yield StreamChunk(text=delta.get("text", ""))
                elif delta.get("type") == "input_json_delta" and d["index"] in index_of:
                    yield StreamChunk(
                        tool_calls=[
                            ToolCallDelta(
                                index=index_of[d["index"]], arguments=delta.get("partial_json", "")
                            )
                        ]
                    )
            elif event == "message_delta":
                u = d.get("usage") or {}
                out = u.get("output_tokens") or 0
                usage = Usage(
                    prompt_tokens=prompt, completion_tokens=out, total_tokens=prompt + out
                )
                yield StreamChunk(
                    finish_reason=_STOP.get((d.get("delta") or {}).get("stop_reason") or "", None),
                    usage=usage,
                    model=model,
                )
            elif event == "error":
                err = d.get("error") or {}
                raise (
                    RateLimited(err.get("message", ""))
                    if err.get("type") == "rate_limit_error"
                    else TransientError(f"{err.get('type')}: {err.get('message')}")
                )

    async def embed(self, texts: list[str], model: str | None = None) -> EmbedResponse:
        raise LLMConfigError("Anthropic has no embeddings API; use another provider for embeddings")
