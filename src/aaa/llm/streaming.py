from __future__ import annotations

from collections.abc import AsyncIterator

from aaa.llm.openai_compat import _parse_tool_call
from aaa.llm.types import LLMResponse, Message, StreamChunk, Usage


async def collect(chunks: AsyncIterator[StreamChunk]) -> LLMResponse:
    """Fold a chunk stream into the same LLMResponse a non-streaming call returns."""
    text: list[str] = []
    calls: dict[int, dict[str, str]] = {}
    usage, finish, model, state_id = Usage(), None, "", None
    async for ch in chunks:
        text.append(ch.text)
        for d in ch.tool_calls:
            c = calls.setdefault(d.index, {"id": "", "name": "", "arguments": ""})
            c["id"] = d.id or c["id"]
            c["name"] = d.name or c["name"]
            c["arguments"] += d.arguments
        usage = ch.usage or usage
        finish = ch.finish_reason or finish
        model = ch.model or model
        state_id = ch.state_id or state_id
    tool_calls = [
        _parse_tool_call(
            {"id": c["id"], "function": {"name": c["name"], "arguments": c["arguments"] or "{}"}}
        )
        for _, c in sorted(calls.items())
    ]
    joined = "".join(text)
    return LLMResponse(
        message=Message(role="assistant", content=joined or None, tool_calls=tool_calls),
        usage=usage,
        model=model,
        finish_reason=finish,
        state_id=state_id,
    )
