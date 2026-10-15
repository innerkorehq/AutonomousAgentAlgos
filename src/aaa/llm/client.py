from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Protocol, runtime_checkable

from aaa.llm.types import EmbedResponse, LLMRequest, LLMResponse, StreamChunk


@runtime_checkable
class LLMClient(Protocol):
    """The only thing algorithms know about LLMs. Any gateway/provider/custom backend that
    can implement this works; errors must be raised as aaa.llm.errors types."""

    async def complete(self, request: LLMRequest, /) -> LLMResponse: ...

    async def embed(self, texts: list[str], model: str | None = None) -> EmbedResponse: ...

    async def aclose(self) -> None: ...


@runtime_checkable
class StreamingLLMClient(Protocol):
    """Optional capability: incremental output. Clients without it still work; the handler
    falls back to a single completion."""

    def stream(self, request: LLMRequest, /) -> AsyncIterator[StreamChunk]: ...


@runtime_checkable
class ServerStateClient(Protocol):
    """Optional capability: the PROVIDER keeps the conversation (OpenAI's Responses API with `store`,
    and similar). A client with it sets `server_state = True`, returns `LLMResponse.state_id`, accepts
    `LLMRequest.continue_from` (then `messages` holds only what is new), and can `forget_state`.

    The handler uses it to avoid resending the whole conversation every turn, and falls back to a full
    request whenever the state is missing or the context policy had to rewrite the conversation. The
    cost is that the provider now holds your data: erasing a run deletes its provider-side state too."""

    server_state: bool

    async def forget_state(self, state_ids: Sequence[str]) -> int:
        """Delete provider-held state; returns how many ids were removed."""
        ...
