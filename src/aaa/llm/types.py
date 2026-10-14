from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    raw_arguments: str | None = None  # set when the model emitted invalid JSON


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None  # for role="tool"
    name: str | None = None


class ToolSpec(BaseModel):
    name: str
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=lambda: {"type": "object", "properties": {}})


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class LLMRequest(BaseModel):
    model: str | None = None  # None = client default
    messages: list[Message]
    tools: list[ToolSpec] = Field(default_factory=list)
    response_schema: dict[str, Any] | None = None  # JSON Schema -> structured output
    params: dict[str, Any] = Field(default_factory=dict)  # temperature, max_tokens, ...
    stream: bool = False  # ask the handler to stream (emits aaa.llm.delta events)


class LLMResponse(BaseModel):
    message: Message
    usage: Usage = Field(default_factory=Usage)
    cost: float | None = None
    model: str = ""
    finish_reason: str | None = None


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
