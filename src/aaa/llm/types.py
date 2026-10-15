from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_serializer

from aaa.core.parts import Part


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    raw_arguments: str | None = None  # set when the model emitted invalid JSON
    # The provider's own id for this call, when it differs from `id` (AAA makes `id` unique per call).
    # Server-state providers match tool results to calls by THEIR id; None for everyone else.
    provider_id: str | None = None


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None  # for role="tool"
    name: str | None = None
    # Attachments (images, audio, files) that come after `content`, in order. See aaa.llm.parts.
    parts: list[Part] | None = None

    @model_serializer(mode="wrap")
    def _omit_empty_parts(self, handler: Any) -> dict[str, Any]:
        d = handler(self)
        if d.get("parts") is None:
            d.pop(
                "parts", None
            )  # text-only messages serialize exactly as they always did (hashes and logs stay identical)
        return d


class ToolSpec(BaseModel):
    name: str
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=lambda: {"type": "object", "properties": {}})


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = (
        0  # prompt tokens served from the provider's prompt cache (billed at a discount)
    )


class ResumeHint(BaseModel):
    state_id: str
    covered: int  # how many non-system messages of the request that state already contains


class LLMRequest(BaseModel):
    model: str | None = None  # None = client default
    messages: list[Message]
    tools: list[ToolSpec] = Field(default_factory=list)
    response_schema: dict[str, Any] | None = None  # JSON Schema -> structured output
    params: dict[str, Any] = Field(default_factory=dict)  # temperature, max_tokens, ...
    stream: bool = False  # ask the handler to stream (emits aaa.llm.delta events)
    # Provider-held conversation state (see `ServerStateClient`). An algorithm may say "the provider
    # already holds the first `covered` non-system messages of this request as state `state_id`", and the
    # handler then sends only what is new. Purely an optimisation: the request is complete without it.
    resume: ResumeHint | None = None
    # Set by the handler, never by algorithms: continue the provider's state `continue_from`, with
    # `messages` holding only what is new (plus the leading system messages, which become instructions).
    continue_from: str | None = None


class LLMResponse(BaseModel):
    message: Message
    usage: Usage = Field(default_factory=Usage)
    cost: float | None = None
    model: str = ""
    finish_reason: str | None = None
    state_id: str | None = (
        None  # the provider's handle for the conversation up to and including this reply
    )


class EmbedResponse(BaseModel):
    vectors: list[list[float]]
    usage: Usage = Field(default_factory=Usage)
    model: str = ""


class ToolCallDelta(BaseModel):
    index: int
    id: str | None = None
    name: str | None = None
    arguments: str = ""  # JSON fragment


class StreamChunk(BaseModel):
    text: str = ""
    tool_calls: list[ToolCallDelta] = Field(default_factory=list)
    finish_reason: str | None = None
    usage: Usage | None = None
    model: str = ""
    state_id: str | None = (
        None  # final chunk of a server-state provider: the handle for this conversation
    )
