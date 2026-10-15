"""Client for OpenAI's Responses API (`POST /v1/responses`), which can hold the conversation for you.

    client = build_client(LLMConfig(provider="openai_responses", model="gpt-4o", base_url="https://api.openai.com/v1",
                                    api_key="${OPENAI_API_KEY}"))
    LLMHandler(client).attach(engine)     # long runs now send only what is new each turn

With `store=True` (the default) every response is kept by the provider and returned as a `state_id`; the
`LLMHandler` then continues from it (`previous_response_id`) and sends just the new messages, so token cost and
latency stop growing with the conversation. Set `store=False` for stateless use (a normal request each time).

What the provider holds is YOUR data, retained by it under its own policy: `engine.erase(...)` deletes the
state ids it knows about (`DELETE /responses/{id}`), but only if the `LLMHandler` was attached to the engine.
Verified against OpenAI's published API and SDK types; other gateways implement `/responses` to varying degrees."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from typing import Any

import httpx

from aaa.llm.config import LLMConfig, expand_env
from aaa.llm.errors import (
    ContentFiltered,
    FatalError,
    LLMConfigError,
    StateNotFound,
    TransientError,
)
from aaa.llm.openai_compat import classify_http_error
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


def _split_system(messages: Sequence[Message]) -> tuple[list[str], list[Message]]:
    """Leading system messages become `instructions` (the Responses API does not carry them over a
    `previous_response_id`, so they are sent on every request); the rest are input items."""
    i = 0
    while i < len(messages) and messages[i].role == "system":
        i += 1
    return [m.content or "" for m in messages[:i] if m.content], list(messages[i:])


def input_items(messages: Sequence[Message]) -> list[dict[str, Any]]:
    """AAA messages as Responses-API input items. Tool results are matched to calls by the PROVIDER's call id."""
    provider_id = {t.id: (t.provider_id or t.id) for m in messages for t in m.tool_calls}
    items: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "tool":
            cid = m.tool_call_id or ""
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": provider_id.get(cid, cid),
                    "output": m.content or "",
                }
            )
        elif m.role == "assistant":
            if m.content:
                items.append({"role": "assistant", "content": m.content})
            items += [
                {
                    "type": "function_call",
                    "call_id": t.provider_id or t.id,
                    "name": t.name,
                    "arguments": json.dumps(t.arguments),
                }
                for t in m.tool_calls
            ]
        else:  # user, and system messages that are not at the start
            items.append({"role": m.role, "content": m.content or ""})
    return items


class OpenAIResponsesClient:
    def __init__(
        self,
        config: LLMConfig,
        *,
        http_client: httpx.AsyncClient | None = None,
        store: bool = True,
        truncation: str | None = None,
    ) -> None:
        if not config.base_url:
            raise LLMConfigError("base_url is required for openai_responses")
        self.config, self.store, self.truncation = config, store, truncation
        self.server_state = store
        self._headers = {k: expand_env(v) for k, v in config.headers.items()}
        if config.api_key:
            self._headers.setdefault("authorization", f"Bearer {expand_env(config.api_key)}")
        self._headers.setdefault("content-type", "application/json")
        self._http = http_client or httpx.AsyncClient(timeout=config.timeout_s)
        self._owns_http = http_client is None

    def _url(self, path: str) -> str:
        return self.config.base_url.rstrip("/") + path

    # -- request ----------------------------------------------------------------------------
    def _payload(self, request: LLMRequest) -> dict[str, Any]:
        instructions, rest = _split_system(request.messages)
        params = {**self.config.params, **request.params}
        if "max_tokens" in params:
            params["max_output_tokens"] = params.pop("max_tokens")
        payload: dict[str, Any] = {
            **params,
            "model": request.model or self.config.model,
            "input": input_items(rest),
            "store": self.store,
        }
        if instructions:
            payload["instructions"] = "\n\n".join(instructions)
        if request.continue_from:
            payload["previous_response_id"] = request.continue_from
        if self.truncation:
            payload.setdefault("truncation", self.truncation)
        if request.tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                    "strict": False,
                }
                for t in request.tools
            ]
        if request.response_schema:
            payload["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": "output",
                    "schema": request.response_schema,
                    "strict": True,
                }
            }
        return payload

    def _cost(self, u: Usage) -> float | None:
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

    @staticmethod
    def _error(resp: httpx.Response) -> Exception:
        low = resp.text[:600].lower()
        if resp.status_code in (400, 404) and (
            "previous_response_not_found" in low
            or ("previous response" in low and "not found" in low)
        ):
            return StateNotFound(resp.text[:300])
        return classify_http_error(resp)

    async def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            resp = await self._http.post(
                self._url(self.config.responses_path), json=payload, headers=self._headers
            )
        except httpx.TimeoutException as e:
            raise TransientError(f"timeout: {e}") from e
        except httpx.TransportError as e:
            raise TransientError(f"transport error: {e}") from e
        if resp.status_code >= 400:
            raise self._error(resp)
        try:
            return resp.json()
        except ValueError as e:
            raise TransientError(f"non-JSON response: {resp.text[:200]}") from e

    # -- response ---------------------------------------------------------------------------
    @staticmethod
    def _usage(u: dict[str, Any] | None) -> Usage:
        u = u or {}
        return Usage(
            prompt_tokens=u.get("input_tokens", 0),
            completion_tokens=u.get("output_tokens", 0),
            total_tokens=u.get("total_tokens")
            or u.get("input_tokens", 0) + u.get("output_tokens", 0),
            cached_tokens=(u.get("input_tokens_details") or {}).get("cached_tokens") or 0,
        )

    def _parse(self, data: dict[str, Any], model: str) -> LLMResponse:
        if data.get("status") == "failed" or data.get("error"):
            err = data.get("error") or {}
            msg = f"{err.get('code', 'error')}: {err.get('message', 'response failed')}"
            raise (
                ContentFiltered(msg)
                if "content" in str(err.get("code", "")) and "filter" in msg
                else TransientError(msg)
            )
        texts: list[str] = []
        calls: list[ToolCall] = []
        for item in data.get("output") or []:
            kind = item.get("type")
            if kind == "message":
                for part in item.get("content") or []:
                    if part.get("type") == "output_text":
                        texts.append(part.get("text", ""))
                    elif part.get("type") == "refusal":
                        texts.append(part.get("refusal", ""))
            elif kind == "function_call":
                raw = item.get("arguments") or "{}"
                cid = item.get("call_id") or item.get("id") or ""
                try:
                    args = json.loads(raw)
                    if not isinstance(args, dict):
                        raise TypeError
                    calls.append(
                        ToolCall(id=cid, name=item.get("name", ""), arguments=args, provider_id=cid)
                    )
                except (ValueError, TypeError):
                    calls.append(
                        ToolCall(
                            id=cid,
                            name=item.get("name", ""),
                            raw_arguments=str(raw),
                            provider_id=cid,
                        )
                    )
        reason = (data.get("incomplete_details") or {}).get("reason")
        if reason == "content_filter":
            raise ContentFiltered("response blocked by content filter")
        usage = self._usage(data.get("usage"))
        return LLMResponse(
            message=Message(role="assistant", content="".join(texts) or None, tool_calls=calls),
            usage=usage,
            cost=self._cost(usage),
            model=data.get("model", model),
            finish_reason="tool_calls"
            if calls
            else ("length" if reason == "max_output_tokens" else "stop"),
            state_id=data.get("id") if self.store else None,
        )

    async def complete(self, request: LLMRequest, /) -> LLMResponse:
        payload = self._payload(request)
        return self._parse(await self._post(payload), payload["model"])

    async def stream(self, request: LLMRequest, /) -> AsyncIterator[StreamChunk]:
        """Server-sent events. Not retried once started."""
        payload = {**self._payload(request), "stream": True}
        try:
            async with self._http.stream(
                "POST", self._url(self.config.responses_path), json=payload, headers=self._headers
            ) as resp:
                if resp.status_code >= 400:
                    await resp.aread()
                    raise self._error(resp)
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    body = line[5:].strip()
                    if not body or body == "[DONE]":
                        continue
                    try:
                        ev = json.loads(body)
                    except ValueError as e:
                        raise TransientError(f"bad stream frame: {body[:100]}") from e
                    chunk = self._chunk(ev, payload["model"])
                    if chunk is not None:
                        yield chunk
        except httpx.TimeoutException as e:
            raise TransientError(f"timeout: {e}") from e
        except httpx.TransportError as e:
            raise TransientError(f"transport error: {e}") from e

    def _chunk(self, ev: dict[str, Any], model: str) -> StreamChunk | None:
        kind = ev.get("type", "")
        if kind == "response.output_text.delta":
            return StreamChunk(text=ev.get("delta", ""))
        if (
            kind == "response.output_item.added"
            and (ev.get("item") or {}).get("type") == "function_call"
        ):
            item = ev["item"]
            return StreamChunk(
                tool_calls=[
                    ToolCallDelta(
                        index=ev.get("output_index", 0),
                        id=item.get("call_id"),
                        name=item.get("name"),
                    )
                ]
            )
        if kind == "response.function_call_arguments.delta":
            return StreamChunk(
                tool_calls=[
                    ToolCallDelta(index=ev.get("output_index", 0), arguments=ev.get("delta", ""))
                ]
            )
        if kind in ("response.completed", "response.incomplete"):
            r = ev.get("response") or {}
            calls = any(i.get("type") == "function_call" for i in r.get("output") or [])
            reason = (r.get("incomplete_details") or {}).get("reason")
            return StreamChunk(
                usage=self._usage(r.get("usage")),
                finish_reason="tool_calls"
                if calls
                else ("length" if reason == "max_output_tokens" else "stop"),
                model=r.get("model", model),
                state_id=r.get("id") if self.store else None,
            )
        if kind in ("response.failed", "error"):
            err = (ev.get("response") or {}).get("error") or ev.get("error") or {}
            raise TransientError(
                f"{err.get('code', 'error')}: {err.get('message', 'stream failed')}"
            )
        return None

    # -- state ------------------------------------------------------------------------------
    async def forget_state(self, state_ids: Sequence[str]) -> int:
        """Delete stored responses at the provider (`DELETE /responses/{id}`); already-gone ids are fine."""
        n = 0
        for sid in state_ids:
            try:
                resp = await self._http.delete(
                    self._url(f"{self.config.responses_path}/{sid}"), headers=self._headers
                )
            except httpx.HTTPError as e:
                raise TransientError(f"could not delete provider state {sid}: {e}") from e
            if resp.status_code == 404:
                continue
            if resp.status_code >= 400:
                raise FatalError(
                    f"could not delete provider state {sid}: HTTP {resp.status_code} {resp.text[:200]}"
                )
            n += 1
        return n

    async def embed(self, texts: list[str], model: str | None = None) -> EmbedResponse:
        raise NotImplementedError(
            "the Responses API has no embeddings endpoint; use an openai_compat client for embeddings"
        )

    async def aclose(self) -> None:
        if self._owns_http:
            await self._http.aclose()
