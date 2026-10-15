"""Guards that need no outside service: patterns, lists, limits, your own function, or a model of your choice."""

from __future__ import annotations

import hashlib
import inspect
import json
import re
from collections.abc import Awaitable, Callable, Iterable, Sequence
from typing import Any, cast

from aaa.guard.types import STAGES, Action, Finding, GuardContext, GuardResult, Stage
from aaa.llm.client import LLMClient
from aaa.llm.types import LLMRequest, Message


def _stages(stages: Iterable[Stage] | None, default: Iterable[Stage] = STAGES) -> frozenset[Stage]:
    return frozenset(stages if stages is not None else default)


class _Base:
    version = "1"
    deterministic = True

    def __init__(
        self, name: str, stages: Iterable[Stage] | None, default: Iterable[Stage] = STAGES
    ) -> None:
        self.name, self.stages = name, _stages(stages, default)


class RegexGuard(_Base):
    """Match patterns. `action="block"` stops on a match, `"flag"` records it, `"redact"` replaces each match with `replacement`.

    RegexGuard({"secret.aws_key": r"AKIA[0-9A-Z]{16}", "pii.ssn": r"\\b\\d{3}-\\d{2}-\\d{4}\\b"}, action="redact")
    """

    def __init__(
        self,
        patterns: dict[str, str] | Sequence[str],
        *,
        action: Action = "block",
        replacement: str = "[REDACTED]",
        name: str = "regex",
        stages: Iterable[Stage] | None = None,
        flags: int = re.IGNORECASE,
    ) -> None:
        super().__init__(name, stages)
        pats = (
            patterns
            if isinstance(patterns, dict)
            else {f"pattern.{i + 1}": p for i, p in enumerate(patterns)}
        )
        self.rules = [(cat, re.compile(p, flags)) for cat, p in pats.items()]
        self.action: Action = action
        self.replacement = replacement

    async def check(self, content: str, ctx: GuardContext, /) -> GuardResult:
        findings, text = [], content
        for cat, rx in self.rules:
            n = len(rx.findall(content))
            if n:
                findings.append(Finding(cat, detail=f"{n} match(es)"))
                if self.action == "redact":
                    text = rx.sub(self.replacement, text)
        if not findings:
            return GuardResult()
        return GuardResult(self.action, findings, text if self.action == "redact" else None)


class DenyListGuard(_Base):
    """Block (or flag) content containing any listed phrase, case-insensitively, on word boundaries."""

    def __init__(
        self,
        phrases: Sequence[str],
        *,
        category: str = "denied_phrase",
        action: Action = "block",
        name: str = "denylist",
        stages: Iterable[Stage] | None = None,
    ) -> None:
        super().__init__(name, stages)
        self.rx = (
            re.compile(r"\b(?:" + "|".join(re.escape(p) for p in phrases) + r")\b", re.IGNORECASE)
            if phrases
            else None
        )
        self.category = category
        self.action: Action = action

    async def check(self, content: str, ctx: GuardContext, /) -> GuardResult:
        if self.rx and self.rx.search(content):
            return GuardResult(self.action, [Finding(self.category)])
        return GuardResult()


class MaxLengthGuard(_Base):
    """Block content longer than `max_chars` (a cheap defence against context-stuffing and runaway outputs)."""

    def __init__(
        self, max_chars: int, *, name: str = "max_length", stages: Iterable[Stage] | None = None
    ) -> None:
        super().__init__(name, stages)
        self.max_chars = max_chars

    async def check(self, content: str, ctx: GuardContext, /) -> GuardResult:
        if len(content) > self.max_chars:
            return GuardResult(
                "block",
                [Finding("too_long", detail=f"{len(content)} > {self.max_chars} characters")],
            )
        return GuardResult()


class ToolAllowlistGuard(_Base):
    """At the `tool_call` stage: only these tools may be called, or only with arguments `check_args` accepts.

    ToolAllowlistGuard({"search", "read_file"}, escalate={"send_email"})   # send_email needs a human to approve
    """

    def __init__(
        self,
        allowed: Iterable[str],
        *,
        escalate: Iterable[str] = (),
        check_args: Callable[[str, dict[str, Any]], bool] | None = None,
        name: str = "tool_allowlist",
    ) -> None:
        super().__init__(name, ["tool_call"])
        self.allowed, self.escalate, self.check_args = set(allowed), set(escalate), check_args

    async def check(self, content: str, ctx: GuardContext, /) -> GuardResult:
        tool = ctx.tool or ""
        if tool in self.escalate:
            return GuardResult("approve", [Finding("tool_needs_approval", detail=tool)])
        if tool not in self.allowed:
            return GuardResult("block", [Finding("tool_not_allowed", detail=tool)])
        if self.check_args and not self.check_args(tool, ctx.arguments or {}):
            return GuardResult("block", [Finding("tool_arguments_rejected", detail=tool)])
        return GuardResult()


class CallableGuard(_Base):
    """Wrap `fn(content, ctx)` returning a `GuardResult`, an action string, or a bool (False blocks); sync or async."""

    deterministic = False

    def __init__(
        self,
        fn: Callable[[str, GuardContext], Any | Awaitable[Any]],
        *,
        name: str | None = None,
        stages: Iterable[Stage] | None = None,
        deterministic: bool = False,
    ) -> None:
        super().__init__(name or getattr(fn, "__name__", "callable"), stages)
        self.fn, self.deterministic = fn, deterministic

    async def check(self, content: str, ctx: GuardContext, /) -> GuardResult:
        out = self.fn(content, ctx)
        if inspect.isawaitable(out):
            out = await out
        if isinstance(out, GuardResult):
            return out
        if isinstance(out, bool):
            return GuardResult() if out else GuardResult("block", [Finding(self.name)])
        return GuardResult(cast(Action, out), [Finding(self.name)] if out != "allow" else [])


JUDGE_SYSTEM = (
    "You are a safety classifier. Decide whether the text violates any of the listed policies. Reply with JSON only: "
    '{"violation": true|false, "categories": ["<policy name>"], "reason": "<one short sentence, no quotes from the text>"}.'
)
JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "violation": {"type": "boolean"},
        "categories": {"type": "array", "items": {"type": "string"}},
        "reason": {"type": "string"},
    },
    "required": ["violation", "categories"],
}
LLAMA_GUARD_CATEGORIES = {
    "violent_crimes": "Responses that enable, encourage or endorse violent crimes",
    "non_violent_crimes": "Responses that enable, encourage or endorse non-violent crimes",
    "sex_crimes": "Responses that enable, encourage or endorse sex-related crimes",
    "child_exploitation": "Content that sexualises children or abuses them",
    "defamation": "Content that is false and damaging to a real person's reputation",
    "specialized_advice": "Unqualified medical, legal or financial advice presented as authoritative",
    "privacy": "Content exposing sensitive personal information",
    "intellectual_property": "Content that violates intellectual property rights",
    "indiscriminate_weapons": "Content that enables creating chemical, biological, radiological, nuclear or high-yield weapons",
    "hate": "Content demeaning people on protected characteristics",
    "self_harm": "Content that enables or encourages suicide or self-harm",
    "sexual_content": "Erotic content",
    "elections": "Factually incorrect information about elections",
    "code_interpreter_abuse": "Attempts to abuse a code interpreter (exploits, denial of service, container escape)",
}


class LLMJudgeGuard(_Base):
    """A model classifies the text against named policies. Works with any `LLMClient`, so a Llama Guard / Prompt Guard class model
    served through a gateway is `LLMJudgeGuard(client, model="llama-guard-3")`. Its calls are not part of the guarded run's trace or
    budget logic, but their cost is recorded on the verdict (and counted in the run's spend).

    `policies` maps a category name to a one-line description; `LLAMA_GUARD_CATEGORIES` is a ready-made set."""

    deterministic = False

    def __init__(
        self,
        client: LLMClient,
        policies: dict[str, str],
        *,
        model: str | None = None,
        action: Action = "block",
        name: str = "llm_judge",
        stages: Iterable[Stage] | None = None,
        max_chars: int = 20_000,
    ) -> None:
        super().__init__(name, stages)
        self.client, self.policies, self.model, self.max_chars = client, policies, model, max_chars
        self.action: Action = action
        self.version = (
            "1:" + hashlib.sha256(json.dumps(sorted(policies.items())).encode()).hexdigest()[:8]
        )  # not hash(): that differs per process

    async def check(self, content: str, ctx: GuardContext, /) -> GuardResult:
        listing = "\n".join(f"- {k}: {v}" for k, v in self.policies.items())
        resp = await self.client.complete(
            LLMRequest(
                model=self.model,
                messages=[
                    Message(role="system", content=JUDGE_SYSTEM),
                    Message(
                        role="user",
                        content=f"Policies:\n{listing}\n\nText to classify (stage: {ctx.stage}):\n<<<\n{content[: self.max_chars]}\n>>>",
                    ),
                ],
                response_schema=JUDGE_SCHEMA,
                params={"temperature": 0},
            )
        )
        usage = {
            "prompt_tokens": resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens,
            "total_tokens": resp.usage.total_tokens,
        }
        m = re.search(r"\{.*\}", resp.message.content or "", re.DOTALL)
        try:
            data = json.loads(m.group() if m else "")
            violated = bool(data["violation"])
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError(
                "judge reply was not the requested JSON"
            ) from exc  # an unreadable verdict is a guard error
        cats = [c for c in data.get("categories", []) if c in self.policies] or (
            ["policy_violation"] if violated else []
        )
        if not violated:
            return GuardResult(usage=usage, cost=resp.cost or 0.0)
        return GuardResult(
            self.action,
            [Finding(c, detail=str(data.get("reason", ""))[:200]) for c in cats],
            usage=usage,
            cost=resp.cost or 0.0,
        )
