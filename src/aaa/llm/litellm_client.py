from __future__ import annotations

from typing import Any

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
from aaa.llm.openai_compat import _msg_out, _parse_tool_call
from aaa.llm.types import EmbedResponse, LLMRequest, LLMResponse, Message, Usage


class LiteLLMClient:
    """In-process LiteLLM SDK adapter (`pip install autonomous-agent-algos[litellm]`)."""

    def __init__(self, config: LLMConfig) -> None:
        try:
            import litellm
        except ImportError as e:  # pragma: no cover
            raise LLMConfigError("install the 'litellm' extra to use provider='litellm'") from e
        self._litellm = litellm
        self.config = config
        self._kw: dict[str, Any] = {"timeout": config.timeout_s}
        if config.base_url:
            self._kw["api_base"] = config.base_url
        if config.api_key:
            self._kw["api_key"] = expand_env(config.api_key)
        if config.headers:
            self._kw["extra_headers"] = {k: expand_env(v) for k, v in config.headers.items()}

    def _wrap(self, e: Exception) -> Exception:
        n = self._litellm
        table = [
            (n.RateLimitError, RateLimited),
            (n.AuthenticationError, AuthError),
            (n.ContextWindowExceededError, ContextOverflow),
            (n.ContentPolicyViolationError, ContentFiltered),
            (n.Timeout, TransientError),
            (n.APIConnectionError, TransientError),
            (n.ServiceUnavailableError, TransientError),
            (n.InternalServerError, TransientError),
        ]
        for src, dst in table:
            if isinstance(e, src):
                return dst(str(e))
        return FatalError(str(e))

    async def complete(self, request: LLMRequest) -> LLMResponse:
        model = request.model or self.config.model
        kw: dict[str, Any] = {**self._kw, **self.config.params, **request.params}
        if request.tools:
            kw["tools"] = [
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
            kw["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "output",
                    "schema": request.response_schema,
                    "strict": True,
                },
            }
        try:
            r = await self._litellm.acompletion(
                model=model, messages=[_msg_out(m) for m in request.messages], **kw
            )
        except Exception as e:
            raise self._wrap(e) from e
        choice = r.choices[0]
        raw = choice.message.model_dump()
        u = getattr(r, "usage", None)
        try:
            cost = self._litellm.completion_cost(completion_response=r)
        except Exception:  # noqa: BLE001 - cost is best-effort; unknown models have no price
            cost = None
        return LLMResponse(
            message=Message(
                role="assistant",
                content=raw.get("content"),
                tool_calls=[_parse_tool_call(t) for t in raw.get("tool_calls") or []],
            ),
            usage=Usage(
                prompt_tokens=getattr(u, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(u, "completion_tokens", 0) or 0,
                total_tokens=getattr(u, "total_tokens", 0) or 0,
            ),
            cost=cost,
            model=getattr(r, "model", model),
            finish_reason=choice.finish_reason,
        )

    async def embed(self, texts: list[str], model: str | None = None) -> EmbedResponse:
        try:
            r = await self._litellm.aembedding(
                model=model or self.config.model, input=texts, **self._kw
            )
        except Exception as e:
            raise self._wrap(e) from e
        return EmbedResponse(
            vectors=[d["embedding"] for d in r.data], model=getattr(r, "model", "")
        )

    async def aclose(self) -> None:
        return None
