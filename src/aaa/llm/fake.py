from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Iterable

from aaa.events.event import Event
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


def text(content: str, **usage: int) -> LLMResponse:
    return LLMResponse(
        message=Message(role="assistant", content=content),
        usage=Usage(**usage),
        model="fake",
        finish_reason="stop",
    )


def tool_call(name: str, call_id: str = "c1", **arguments) -> LLMResponse:
    return LLMResponse(
        message=Message(
            role="assistant", tool_calls=[ToolCall(id=call_id, name=name, arguments=arguments)]
        ),
        model="fake",
        finish_reason="tool_calls",
    )


class FakeLLMClient:
    """Scripted responses (in order, or via a function of the request) for deterministic tests.
    Exceptions in the script are raised, so error paths are testable too."""

    def __init__(
        self, script: Iterable[LLMResponse | Exception] | Callable[[LLMRequest], LLMResponse]
    ):
        self._fn = script if callable(script) else None
        self._it = None if callable(script) else iter(script)
        self.requests: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        if self._fn:
            return self._fn(request)
        assert self._it is not None
        try:
            item = next(self._it)
        except StopIteration:
            raise AssertionError("FakeLLMClient script exhausted") from None
        if isinstance(item, Exception):
            raise item
        return item

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:
        """Replays the scripted response as word-sized chunks (and tool-call deltas)."""
        r = await self.complete(request)
        words = (r.message.content or "").split(" ")
        for i, w in enumerate(words if r.message.content else []):
            yield StreamChunk(text=w + (" " if i < len(words) - 1 else ""))
        for i, t in enumerate(r.message.tool_calls):
            import json

            raw = json.dumps(t.arguments)
            yield StreamChunk(
                tool_calls=[ToolCallDelta(index=i, id=t.id, name=t.name, arguments=raw[:3])]
            )
            yield StreamChunk(tool_calls=[ToolCallDelta(index=i, arguments=raw[3:])])
        yield StreamChunk(usage=r.usage, finish_reason=r.finish_reason, model=r.model)

    async def embed(self, texts: list[str], model: str | None = None) -> EmbedResponse:
        return EmbedResponse(vectors=[[float(len(t))] for t in texts], model="fake")

    async def aclose(self) -> None:
        return None


class ReplayLLMClient(FakeLLMClient):
    """Serve the responses recorded in a run's `aaa.llm.completed` events, in order, for
    deterministic golden-test replays of a real run."""

    def __init__(self, events: Iterable[Event]):
        super().__init__(
            LLMResponse.model_validate(e.data["response"])
            for e in events
            if e.type == "aaa.llm.completed"
        )
