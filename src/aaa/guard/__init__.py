"""Guardrails: check what goes into the model, what comes out, what it asks tools to do, and what tools hand back.

    policy = GuardPolicy([
        RegexGuard({"pii.ssn": r"\\b\\d{3}-\\d{2}-\\d{4}\\b"}, action="redact"),
        LLMJudgeGuard(judge_client, LLAMA_GUARD_CATEGORIES, stages=["input", "tool_result"]),
        ToolAllowlistGuard({"search"}, escalate={"send_email"}),
    ])
    LLMHandler(client, guards=policy).attach(engine)
    ToolHandler(registry, guards=policy).attach(engine)

Enforcement is inline (a block ends the run as `OutcomeKind.BLOCKED`; a blocked tool call is a normal tool failure; a poisoned tool
result is replaced by a placeholder), and every check leaves an `aaa.guard.verdict` event. Guardrails reduce risk; they are not a
security boundary."""

from aaa.guard.builtin import (
    LLAMA_GUARD_CATEGORIES,
    CallableGuard,
    DenyListGuard,
    LLMJudgeGuard,
    MaxLengthGuard,
    RegexGuard,
    ToolAllowlistGuard,
)
from aaa.guard.policy import Decision, GuardPolicy, GuardSpec, Verdict
from aaa.guard.types import (
    Action,
    Finding,
    GuardContext,
    Guardrail,
    GuardResult,
    Stage,
)
from aaa.guard.watch import GuardWatcher
from aaa.llm.errors import GuardrailBlocked

__all__ = [
    "LLAMA_GUARD_CATEGORIES",
    "Action",
    "CallableGuard",
    "Decision",
    "DenyListGuard",
    "Finding",
    "GuardContext",
    "GuardPolicy",
    "GuardResult",
    "GuardSpec",
    "GuardWatcher",
    "Guardrail",
    "GuardrailBlocked",
    "LLMJudgeGuard",
    "MaxLengthGuard",
    "RegexGuard",
    "Stage",
    "ToolAllowlistGuard",
    "Verdict",
]
