"""Token counting for context budgeting.

Exact counts need the provider's tokenizer, which differs per model and is often unavailable. Context
management does not need exactness: it needs an estimate that is cheap, deterministic and *calibrated*
against what the provider actually reports (`usage.prompt_tokens`). `HeuristicCounter` is the estimate,
`CalibratedCounter` learns the correction factor from real responses, and `TiktokenCounter` is there if
you want a real tokenizer for OpenAI-family models."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from aaa.llm.types import Message, ToolSpec


@runtime_checkable
class TokenCounter(Protocol):
    def count(self, messages: Sequence[Message], tools: Sequence[ToolSpec] = ()) -> int: ...


def message_text(m: Message) -> str:
    """Everything in a message that the provider will tokenize."""
    parts = [m.content or ""]
    parts += [
        f"{c.name}{json.dumps(c.arguments, separators=(',', ':'), default=str)}"
        for c in m.tool_calls
    ]
    return "\n".join(p for p in parts if p)


class HeuristicCounter:
    """`chars / chars_per_token` plus a fixed per-message overhead. The default of 3.5 characters per token errs
    on the high side for English prose and is about right for code and JSON, so it over-counts a little: the
    safe direction for a budget."""

    def __init__(
        self, chars_per_token: float = 3.5, per_message: int = 4, per_tool_call: int = 8
    ) -> None:
        self.cpt, self.per_message, self.per_tool_call = chars_per_token, per_message, per_tool_call

    def text(self, s: str) -> int:
        return math.ceil(len(s) / self.cpt) if s else 0

    def count(self, messages: Sequence[Message], tools: Sequence[ToolSpec] = ()) -> int:
        n = sum(
            self.per_message + self.text(message_text(m)) + self.per_tool_call * len(m.tool_calls)
            for m in messages
        )
        if tools:
            n += self.text(
                json.dumps([t.model_dump() for t in tools], separators=(",", ":"), default=str)
            )
        return n


class TiktokenCounter:
    """Exact counts for OpenAI-family models via `tiktoken` (`pip install tiktoken`). Other providers' models
    tokenize differently, so pair it with `CalibratedCounter` rather than trusting it blindly."""

    def __init__(self, encoding: str = "o200k_base", per_message: int = 4) -> None:
        import tiktoken  # pyright: ignore[reportMissingImports]

        self.enc, self.per_message = tiktoken.get_encoding(encoding), per_message

    def count(self, messages: Sequence[Message], tools: Sequence[ToolSpec] = ()) -> int:
        n = sum(self.per_message + len(self.enc.encode(message_text(m))) for m in messages)
        if tools:
            n += len(
                self.enc.encode(
                    json.dumps([t.model_dump() for t in tools], separators=(",", ":"), default=str)
                )
            )
        return n


class CalibratedCounter:
    """Wraps a counter and scales it by the ratio `actual / estimated` seen on real responses (a smoothed
    average, per model), so a systematic error in the estimate stops mattering after a few calls."""

    def __init__(
        self,
        base: TokenCounter | None = None,
        *,
        smoothing: float = 0.3,
        bounds: tuple[float, float] = (0.4, 3.0),
    ) -> None:
        self.base = base or HeuristicCounter()
        self.smoothing, self.bounds = smoothing, bounds
        self._ratio: dict[str | None, float] = {}
        self.model: str | None = None  # the model the next count is for (set by the policy)

    def ratio(self, model: str | None = None) -> float:
        return self._ratio.get(model if model is not None else self.model, 1.0)

    def count(self, messages: Sequence[Message], tools: Sequence[ToolSpec] = ()) -> int:
        return math.ceil(self.base.count(messages, tools) * self.ratio())

    def observe(self, model: str | None, estimated: int, actual: int) -> None:
        """Feed back one real measurement: the estimate we made for a request and the provider's own count.
        `estimated` must be the *uncalibrated* estimate (`base.count`) of exactly what was sent."""
        if estimated <= 0 or actual <= 0:
            return
        lo, hi = self.bounds
        seen = min(hi, max(lo, actual / estimated))
        old = self._ratio.get(model)
        self._ratio[model] = (
            seen if old is None else (1 - self.smoothing) * old + self.smoothing * seen
        )


def as_dict(m: Message) -> dict[str, Any]:
    return m.model_dump(mode="json")
