from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Iterable, Sequence

from aaa.events.event import Event
from aaa.llm.errors import FatalError, StateNotFound
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


class FakeStatefulClient:
    """Plays a provider that HOLDS conversations (like the OpenAI Responses API), for tests and demos.

    `script(request)` sees the FULL conversation as the provider reconstructs it (state + new messages), so it is
    written exactly like a `FakeLLMClient` script. What the handler actually sent is recorded in `sent` as
    `(messages_sent, continue_from)`, so a test can assert on the saving. Tool results are matched to calls by the
    provider's own ids, as the real API does, and an unknown id is a hard error: that is what catches id mix-ups.
    `lose_state_after=n` makes the provider forget every stored state after n calls (expiry, deletion)."""

    server_state = True

    def __init__(
        self, script: Callable[[LLMRequest], LLMResponse], *, lose_state_after: int | None = None
    ) -> None:
        self._fn = script
        self.states: dict[str, list[Message]] = {}
        self.sent: list[tuple[int, str | None]] = []
        self.deleted: list[str] = []
        self.lose_after = lose_state_after
        self._n = 0
        self._calls = 0

    async def complete(self, request: LLMRequest, /) -> LLMResponse:
        self._calls += 1
        if self.lose_after is not None and self._calls == self.lose_after + 1:
            self.states.clear()  # the provider's state expires
        self.sent.append((len(request.messages), request.continue_from))
        systems = [m for m in request.messages if m.role == "system"]
        rest = [m for m in request.messages if m.role != "system"]
        base: list[Message] = []
        if request.continue_from:
            if request.continue_from not in self.states:
                raise StateNotFound(f"previous response {request.continue_from} not found")
            base = [m for m in self.states[request.continue_from] if m.role != "system"]
        known = {c.provider_id or c.id for m in [*base, *rest] for c in m.tool_calls}
        to_provider = {c.id: c.provider_id or c.id for m in rest for c in m.tool_calls}
        fixed: list[Message] = []
        for m in rest:
            if m.role == "tool":
                cid = to_provider.get(m.tool_call_id or "", m.tool_call_id or "")
                if cid not in known:
                    raise FatalError(f"tool result for unknown call id {m.tool_call_id!r}")
                m = m.model_copy(update={"tool_call_id": cid})
            elif m.tool_calls:
                m = m.model_copy(
                    update={
                        "tool_calls": [
                            c.model_copy(update={"id": c.provider_id or c.id}) for c in m.tool_calls
                        ]
                    }
                )
            fixed.append(m)
        full = [*systems, *base, *fixed]
        resp = self._fn(request.model_copy(update={"messages": full, "continue_from": None}))
        calls = []
        for c in resp.message.tool_calls:  # a real provider gives every call a unique id
            self._n += 1
            calls.append(
                c.model_copy(update={"id": f"prov_{self._n}", "provider_id": f"prov_{self._n}"})
            )
        reply = resp.message.model_copy(update={"tool_calls": calls})
        self._n += 1
        state = f"state_{self._n}"
        self.states[state] = [*full, reply]
        return resp.model_copy(update={"message": reply, "state_id": state})

    async def forget_state(self, state_ids: Sequence[str]) -> int:
        n = 0
        for sid in state_ids:
            if self.states.pop(sid, None) is not None:
                n += 1
        self.deleted += list(state_ids)
        return n

    async def embed(self, texts: list[str], model: str | None = None) -> EmbedResponse:
        return EmbedResponse(vectors=[[float(len(t))] for t in texts], model="fake")

    async def aclose(self) -> None:
        return None
