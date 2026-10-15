"""Native Google Gemini client (`provider="gemini"`, the Generative Language API), over httpx: no vendor SDK needed.

    LLMConfig(provider="gemini", model="gemini-2.5-flash", api_key="${GEMINI_API_KEY}")

What it maps:

* system messages -> `systemInstruction`; roles are `user` / `model`; consecutive turns of one role are merged;
* tools -> `functionDeclarations` (full JSON Schema via `parametersJsonSchema`); `functionCall` / `functionResponse` parts, with results ordered like the calls
  that produced them (Gemini pairs by name and order). If the API supplies a call `id` it is kept and echoed;
* `response_schema` -> `responseMimeType: application/json` + `responseJsonSchema`;
* attachments: images, audio and files as `inlineData` (bytes) or `fileData` (URL);
* a prompt or reply blocked for safety raises `ContentFiltered`; `thought` parts are not returned as text;
* usage: `prompt_tokens`, `completion_tokens` (thinking tokens are counted as output), `cached_tokens` from `cachedContentTokenCount`;
* streaming (`:streamGenerateContent?alt=sse`) and embeddings (`:batchEmbedContents`).

The API key is sent in the `x-goog-api-key` header (never in the URL). Parameters Gemini does not know are dropped with a warning log."""

from __future__ import annotations

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

_GENERATION = {
    "temperature": "temperature",
    "top_p": "topP",
    "top_k": "topK",
    "max_tokens": "maxOutputTokens",
    "stop": "stopSequences",
    "seed": "seed",
    "n": "candidateCount",
    "frequency_penalty": "frequencyPenalty",
    "presence_penalty": "presencePenalty",
}
_BLOCKED = {
    "SAFETY",
    "BLOCKLIST",
    "PROHIBITED_CONTENT",
    "SPII",
    "RECITATION",
    "IMAGE_SAFETY",
    "IMAGE_PROHIBITED_CONTENT",
    "LANGUAGE",
}
_FINISH = {"STOP": "stop", "MAX_TOKENS": "length"}
_MODES = {"auto": "AUTO", "required": "ANY", "any": "ANY", "none": "NONE"}


def classify_error(resp: httpx.Response) -> Exception:
    """Gemini errors are `{"error": {"code", "message", "status"}}`."""
    try:
        err = resp.json().get("error") or {}
    except ValueError:
        err = {}
    status, msg, code = (
        err.get("status", ""),
        err.get("message") or resp.text[:300],
        resp.status_code,
    )
    if code == 429 or status == "RESOURCE_EXHAUSTED":
        ra = resp.headers.get("retry-after")
        try:
            return RateLimited(msg, float(ra) if ra else None)
        except ValueError:
            return RateLimited(msg)
    if code in (500, 502, 503, 504) or status in ("INTERNAL", "UNAVAILABLE", "DEADLINE_EXCEEDED"):
        return TransientError(f"HTTP {code} {status}: {msg}")
    if code in (401, 403) or status in ("UNAUTHENTICATED", "PERMISSION_DENIED"):
        return AuthError(f"HTTP {code} {status}: {msg}")
    if code == 400 and any(
        s in msg.lower()
        for s in ("token count", "too long", "exceeds the maximum", "context window")
    ):
        return ContextOverflow(msg)
    if code == 404:
        return FatalError(f"HTTP 404 {status}: {msg}")
    return classify_http_error(resp)


def _media_part(p: MediaPart) -> dict[str, Any]:
    if p.url:
        return {"fileData": {"mimeType": p.media_type, "fileUri": p.url}}
    return {"inlineData": {"mimeType": p.media_type, "data": p.data}}


def to_gemini(messages: list[Message]) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """(`systemInstruction`, `contents`)."""
    system: list[str] = []
    turns: list[dict[str, Any]] = []
    pending: dict[
        str, tuple[str, int, str | None]
    ] = {}  # call id -> (tool name, order, provider id)

    def add(role: str, parts: list[dict[str, Any]]) -> None:
        if not parts:
            return
        if turns and turns[-1]["role"] == role:
            turns[-1]["parts"].extend(parts)
        else:
            turns.append({"role": role, "parts": list(parts)})

    for m in messages:
        if m.role == "system":
            if m.content:
                system.append(m.content)
        elif m.role == "assistant":
            parts: list[dict[str, Any]] = [{"text": m.content}] if m.content else []
            pending = {}
            for i, t in enumerate(m.tool_calls):
                call: dict[str, Any] = {"name": t.name, "args": t.arguments}
                if t.provider_id:
                    call["id"] = t.provider_id
                pending[t.id] = (t.name, i, t.provider_id)
                parts.append({"functionCall": call})
            add("model", parts)
        elif m.role == "tool":
            name, order, pid = pending.get(m.tool_call_id or "", (m.name or "tool", 10**6, None))
            resp: dict[str, Any] = {"name": name, "response": {"output": m.content or ""}}
            if pid:
                resp["id"] = pid
            add("user", [{"functionResponse": resp, "_order": order}])
        else:
            parts = [{"text": m.content}] if m.content else []
            parts += [_media_part(p) for p in m.parts or []]
            add("user", parts)
    for t in turns:  # function responses in the order of the calls; then drop the helper key
        resp_parts = [p for p in t["parts"] if "functionResponse" in p]
        if resp_parts:
            others = [p for p in t["parts"] if "functionResponse" not in p]
            t["parts"] = sorted(resp_parts, key=lambda p: p["_order"]) + others
        for p in t["parts"]:
            p.pop("_order", None)
    return ({"parts": [{"text": "\n\n".join(system)}]} if system else None), turns


class GeminiClient(HTTPClientBase):
    def __init__(self, config: LLMConfig, *, http_client: httpx.AsyncClient | None = None) -> None:
        super().__init__(
            config,
            http_client=http_client,
            base_url="https://generativelanguage.googleapis.com/v1beta",
            auth_header="x-goog-api-key",
            default_modalities=("text", "image", "audio", "file"),
        )

    @staticmethod
    def _model(name: str) -> str:
        return name.removeprefix("models/")

    def _payload(self, request: LLMRequest) -> dict[str, Any]:
        system, contents = to_gemini(request.messages)
        params = {**self.config.params, **request.params}
        tool_choice = params.pop("tool_choice", None)
        unknown = sorted(set(params) - set(_GENERATION) - {"stop_sequences"})
        if unknown:
            log.warning(
                "gemini: dropping parameters the Generative Language API does not take: %s", unknown
            )
        gen: dict[str, Any] = {}
        for k, v in params.items():
            if k in _GENERATION:
                gen[_GENERATION[k]] = [v] if k == "stop" and isinstance(v, str) else v
            elif k == "stop_sequences":
                gen["stopSequences"] = v
        if request.response_schema:
            gen["responseMimeType"] = "application/json"
            gen["responseJsonSchema"] = request.response_schema
        payload: dict[str, Any] = {"contents": contents}
        if system:
            payload["systemInstruction"] = system
        if gen:
            payload["generationConfig"] = gen
        if request.tools:
            payload["tools"] = [
                {
                    "functionDeclarations": [
                        {
                            "name": t.name,
                            "description": t.description,
                            "parametersJsonSchema": t.parameters,
                        }
                        for t in request.tools
                    ]
                }
            ]
        if isinstance(tool_choice, str) and tool_choice in _MODES:
            payload["toolConfig"] = {"functionCallingConfig": {"mode": _MODES[tool_choice]}}
        elif isinstance(tool_choice, dict) and (
            name := (tool_choice.get("function") or {}).get("name")
        ):
            payload["toolConfig"] = {
                "functionCallingConfig": {"mode": "ANY", "allowedFunctionNames": [name]}
            }
        return payload

    def _path(self, request: LLMRequest, action: str) -> str:
        return f"/models/{self._model(request.model or self.config.model)}:{action}"

    @staticmethod
    def _usage(u: dict[str, Any]) -> Usage:
        prompt = u.get("promptTokenCount") or 0
        out = (u.get("candidatesTokenCount") or 0) + (u.get("thoughtsTokenCount") or 0)
        return Usage(
            prompt_tokens=prompt,
            completion_tokens=out,
            total_tokens=u.get("totalTokenCount") or prompt + out,
            cached_tokens=u.get("cachedContentTokenCount") or 0,
        )

    def _parse(self, data: dict[str, Any], model: str) -> LLMResponse:
        cands = data.get("candidates") or []
        if not cands:
            block = (data.get("promptFeedback") or {}).get("blockReason")
            if block:
                raise ContentFiltered(f"the prompt was blocked ({block})")
            raise TransientError(f"response has no candidates: {str(data)[:200]}")
        c = cands[0]
        texts: list[str] = []
        calls: list[ToolCall] = []
        for p in (c.get("content") or {}).get("parts") or []:
            if "functionCall" in p:
                fc = p["functionCall"]
                calls.append(
                    ToolCall(
                        id=fc.get("id") or f"call_{len(calls)}",
                        name=fc.get("name", ""),
                        arguments=fc.get("args") or {},
                        provider_id=fc.get("id"),
                    )
                )
            elif p.get("text") is not None and not p.get("thought"):
                texts.append(p["text"])
        reason = c.get("finishReason")
        if reason in _BLOCKED and not texts and not calls:
            raise ContentFiltered(f"the reply was blocked ({reason})")
        if reason == "MALFORMED_FUNCTION_CALL":
            raise TransientError("the model produced a malformed function call")
        usage = self._usage(data.get("usageMetadata") or {})
        finish = "tool_calls" if calls else _FINISH.get(reason or "", reason)
        return LLMResponse(
            message=Message(role="assistant", content="".join(texts) or None, tool_calls=calls),
            usage=usage,
            cost=self.cost(usage),
            model=data.get("modelVersion") or model,
            finish_reason=finish,
        )

    async def complete(self, request: LLMRequest) -> LLMResponse:
        data = await self.post_json(
            self._path(request, "generateContent"), self._payload(request), classify_error
        )
        return self._parse(data, request.model or self.config.model)

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:
        n_calls = 0
        async for _event, d in self.sse(
            self._path(request, "streamGenerateContent") + "?alt=sse",
            self._payload(request),
            classify_error,
        ):
            cands = d.get("candidates") or []
            if not cands:
                if block := (d.get("promptFeedback") or {}).get("blockReason"):
                    raise ContentFiltered(f"the prompt was blocked ({block})")
                continue
            c = cands[0]
            text, deltas = "", []
            for p in (c.get("content") or {}).get("parts") or []:
                if "functionCall" in p:
                    fc = p["functionCall"]
                    deltas.append(
                        ToolCallDelta(
                            index=n_calls,
                            id=fc.get("id") or f"call_{n_calls}",
                            name=fc.get("name"),
                            arguments=json.dumps(fc.get("args") or {}),
                        )
                    )
                    n_calls += 1
                elif p.get("text") and not p.get("thought"):
                    text += p["text"]
            reason = c.get("finishReason")
            if reason in _BLOCKED and not text and not deltas and not n_calls:
                raise ContentFiltered(f"the reply was blocked ({reason})")
            um = d.get("usageMetadata")
            yield StreamChunk(
                text=text,
                tool_calls=deltas,
                finish_reason=("tool_calls" if n_calls else _FINISH.get(reason or "", reason))
                if reason
                else None,
                usage=self._usage(um) if um and reason else None,
                model=d.get("modelVersion", ""),
            )

    async def embed(self, texts: list[str], model: str | None = None) -> EmbedResponse:
        name = self._model(model or self.config.model)
        body = {
            "requests": [
                {"model": f"models/{name}", "content": {"parts": [{"text": t}]}} for t in texts
            ]
        }
        data = await self.post_json(f"/models/{name}:batchEmbedContents", body, classify_error)
        try:
            vectors = [e["values"] for e in data["embeddings"]]
        except (KeyError, TypeError) as e:
            raise TransientError("malformed embeddings response") from e
        return EmbedResponse(vectors=vectors, model=name)
