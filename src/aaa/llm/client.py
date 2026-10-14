from __future__ import annotations

from collections.abc import AsyncIterator
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
