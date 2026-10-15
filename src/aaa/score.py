"""One result vocabulary for evals, guardrail verdicts and trace sinks.

A `Score` is a named judgement of something that happened: a run, a model call, a tool call. Evals produce them,
guardrails produce them (as verdicts), and trace sinks that support it (`Scorable`) attach them to the trace of the
run they are about, so a score shows up next to the spans in Langfuse, LangSmith or Phoenix."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field


class Score(BaseModel):
    name: str
    value: (
        float | bool | str
    )  # numbers are 0..1 by convention; booleans are pass/fail; strings are labels
    reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    # what producing the score cost (an LLM judge, a hosted classifier): counted in the run's budget ledger
    usage: dict[str, int] = Field(default_factory=dict)
    cost: float = 0.0

    @property
    def numeric(self) -> float | None:
        """The score as a number if it is one (bool -> 0/1), else None (a label)."""
        if isinstance(self.value, bool):
            return float(self.value)
        return float(self.value) if isinstance(self.value, int | float) else None

    @property
    def passed(self) -> bool | None:
        """True/False if the score is pass/fail-like (a bool, or a number against the 0.5 midpoint), else None."""
        if isinstance(self.value, bool):
            return self.value
        n = self.numeric
        return None if n is None else n >= 0.5


@runtime_checkable
class Scorable(Protocol):
    """Optional capability of a trace sink: attach scores to a trace (and optionally to one span in it)."""

    async def score(
        self, trace_id: str, scores: Sequence[Score], *, span_id: str | None = None
    ) -> None: ...
