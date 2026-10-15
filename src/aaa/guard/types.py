"""Guardrail vocabulary: a guard looks at text at a stage and returns an action plus findings."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

Stage = Literal["input", "output", "tool_call", "tool_result"]
Action = Literal["allow", "flag", "redact", "approve", "block"]
STAGES: tuple[Stage, ...] = ("input", "output", "tool_call", "tool_result")
# how strong each action is; a policy takes the strongest result
SEVERITY: dict[str, int] = {"allow": 0, "flag": 1, "redact": 2, "approve": 3, "block": 4}


@dataclass(frozen=True)
class Finding:
    """One thing a guard noticed. `category` is a stable machine name (`pii.email`, `prompt_injection`, `violence`)."""

    category: str
    score: float | None = None  # the guard's confidence/severity, if it has one
    detail: str | None = None  # a short human note; NEVER the matched content


@dataclass
class GuardResult:
    action: Action = "allow"
    findings: list[Finding] = field(default_factory=list)
    text: str | None = None  # the redacted text, for action="redact"
    usage: dict[str, int] = field(
        default_factory=dict
    )  # what the check cost (an LLM judge, a hosted classifier)
    cost: float = 0.0

    @property
    def categories(self) -> list[str]:
        return sorted({f.category for f in self.findings})


@dataclass(frozen=True)
class GuardContext:
    stage: Stage
    run_id: str | None = None
    tenant: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    tool: str | None = None  # at the tool stages: which tool
    arguments: dict[str, Any] | None = None  # at tool_call: the parsed arguments
    has_media: bool = False  # the message carries attachments (images/audio/files) that this text-only guard cannot read


@runtime_checkable
class Guardrail(Protocol):
    """Anything that can judge a piece of text.

    * `stages`: where it applies (a policy may narrow this per guard).
    * `version`: bump it when the guard's behaviour changes; stored verdicts are keyed by it.
    * `deterministic`: True for pure functions of the text (regex, deny-list); the policy then never needs a stored verdict to
      stay consistent across a redelivery. False (the default) for models and hosted services.
    """

    name: str
    stages: frozenset[Stage]
    version: str
    deterministic: bool

    async def check(self, content: str, ctx: GuardContext, /) -> GuardResult: ...
