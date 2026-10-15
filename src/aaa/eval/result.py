"""What running one case produced."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from aaa.core.outcome import Outcome
from aaa.eval.dataset import Case
from aaa.score import Score
from aaa.trace import A, Span


def default_extract(value: Any) -> str:
    """The text of an outcome value: a string as is; a dict's answer/text/result/output; else JSON."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for k in ("answer", "text", "result", "output", "final"):
            if isinstance(value.get(k), str):
                return value[k]
    return json.dumps(value, default=str, sort_keys=True)


@dataclass
class CaseResult:
    case: Case
    outcome: Outcome[Any] | None
    text: str = ""
    run_id: str | None = None
    spans: list[Span] = field(default_factory=list)
    steps: int = 0
    tokens: int = 0
    cost: float = 0.0
    latency_s: float = 0.0  # wall clock
    error: str | None = None  # the harness failed to run the case (not: the agent failed)
    repeat: int = 0
    scores: list[Score] = field(default_factory=list)
    turns: list[tuple[str, str]] = field(
        default_factory=list
    )  # (user, assistant) pairs of a multi-turn case

    @property
    def ok(self) -> bool:
        return self.outcome is not None and self.outcome.ok

    @property
    def tool_calls(self) -> list[str]:
        return [str(s.attributes.get(A.TOOL_NAME)) for s in self.spans if s.kind == "tool"]
