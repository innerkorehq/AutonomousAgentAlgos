"""A policy: which guards run at which stage, how they combine, and what happens when one fails."""

from __future__ import annotations

import asyncio
import hashlib
import logging
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal

from aaa.guard.types import SEVERITY, Finding, GuardContext, Guardrail, GuardResult, Stage

log = logging.getLogger("aaa.guard")


@dataclass
class GuardSpec:
    """A guard plus how the policy treats it.

    * `mode="enforce"`: its action is applied inline. `"observe"`: it runs out of line (see `GuardWatcher`), is recorded, and
      never changes the run.
    * `on_error`: what a crash or timeout of the guard means. `"closed"` blocks (safe, but an outage of the guard stops traffic);
      `"open"` lets the content through (available, but unprotected).
    """

    guard: Guardrail
    stages: frozenset[Stage] | None = None  # None = the guard's own
    mode: Literal["enforce", "observe"] = "enforce"
    on_error: Literal["closed", "open"] = "closed"
    timeout_s: float = 10.0

    @property
    def name(self) -> str:
        return self.guard.name

    def applies(self, stage: Stage) -> bool:
        return stage in (self.stages if self.stages is not None else self.guard.stages)


@dataclass
class Verdict:
    guard: str
    version: str
    stage: Stage
    mode: str
    result: GuardResult
    error: str | None = None  # the guard itself failed (the result is then the on_error fallback)
    reused: bool = False  # taken from the log instead of calling the guard again
    content_hash: str = ""

    def data(self) -> dict[str, object]:
        return {
            "guard": self.guard,
            "version": self.version,
            "stage": self.stage,
            "mode": self.mode,
            "action": self.result.action,
            "categories": self.result.categories,
            "scores": {f.category: f.score for f in self.result.findings if f.score is not None},
            "hash": self.content_hash,
            "error": self.error,
            "usage": self.result.usage,
            "cost": self.result.cost,
        }


@dataclass
class Decision:
    action: str = "allow"
    text: str | None = None  # the content to use instead, when redacted
    verdicts: list[Verdict] = field(default_factory=list)
    blocked_by: str | None = None

    @property
    def categories(self) -> list[str]:
        out: set[str] = set()
        for v in self.verdicts:
            if v.result.action != "allow" and v.mode == "enforce":
                out |= set(v.result.categories)
        return sorted(out)


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


Runner = Callable[["GuardSpec", str, GuardContext], Awaitable[Verdict]]


class GuardPolicy:
    """`GuardPolicy([RegexGuard(...), OpenAIModerationGuard(...)])` passed to `LLMHandler(guards=)` / `ToolHandler(guards=)`.

    Enforcing guards at a stage run concurrently on the same content and the strongest action wins
    (`block` > `approve` > `redact` > `flag` > `allow`). Redactions are then applied one after another, so two redacting guards
    compose instead of one overwriting the other."""

    def __init__(
        self,
        guards: Sequence[Guardrail | GuardSpec],
        *,
        on_error: Literal["closed", "open"] = "closed",
        timeout_s: float = 10.0,
    ) -> None:
        self.specs = [
            g if isinstance(g, GuardSpec) else GuardSpec(g, on_error=on_error, timeout_s=timeout_s)
            for g in guards
        ]
        names = [s.name for s in self.specs]
        if len(set(names)) != len(names):
            raise ValueError(
                f"guard names must be unique: {sorted({n for n in names if names.count(n) > 1})}"
            )

    def specs_for(self, stage: Stage, mode: str = "enforce") -> list[GuardSpec]:
        return [s for s in self.specs if s.mode == mode and s.applies(stage)]

    def enforces(self, stage: Stage) -> bool:
        return bool(self.specs_for(stage, "enforce"))

    async def run_one(self, spec: GuardSpec, content: str, ctx: GuardContext) -> Verdict:
        h = content_hash(content + ("\0media" if ctx.has_media else ""))
        try:
            result = await asyncio.wait_for(spec.guard.check(content, ctx), spec.timeout_s)
            return Verdict(
                spec.name, spec.guard.version, ctx.stage, spec.mode, result, content_hash=h
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning(
                "guard %s failed at %s (%s)",
                spec.name,
                ctx.stage,
                type(exc).__name__,
                exc_info=True,
            )
            msg = f"{type(exc).__name__}: {exc}"[:200]
            fallback = (
                GuardResult("block", [Finding("guard_error", detail=type(exc).__name__)])
                if spec.on_error == "closed"
                else GuardResult("allow")
            )
            return Verdict(
                spec.name,
                spec.guard.version,
                ctx.stage,
                spec.mode,
                fallback,
                error=msg,
                content_hash=h,
            )

    async def evaluate(
        self,
        stage: Stage,
        content: str,
        ctx: GuardContext,
        *,
        runner: Runner | None = None,
        mode: str = "enforce",
    ) -> Decision:
        """Run the guards of `stage` and fold their verdicts into one decision. `runner` lets the engine integration reuse
        stored verdicts and record new ones; the default just calls the guards."""
        run = runner or self.run_one
        specs = self.specs_for(stage, mode)
        if not specs:
            return Decision()
        verdicts = list(await asyncio.gather(*(run(s, content, ctx) for s in specs)))
        worst = max(verdicts, key=lambda v: SEVERITY[v.result.action])
        dec = Decision(action=worst.result.action, verdicts=verdicts)
        if dec.action == "block":
            dec.blocked_by = worst.guard
            return dec
        if dec.action == "approve":
            return dec
        redactors = [
            (s, v)
            for s, v in zip(specs, verdicts, strict=True)
            if v.result.action == "redact" and v.result.text is not None
        ]
        if redactors:
            text = redactors[0][1].result.text
            assert text is not None
            for spec, _ in redactors[1:]:  # later redactors must see the text the earlier ones left
                again = await run(spec, text, ctx)
                dec.verdicts.append(again)
                if again.result.action == "block":
                    dec.action, dec.blocked_by = "block", again.guard
                    return dec
                if again.result.action == "redact" and again.result.text is not None:
                    text = again.result.text
            dec.action, dec.text = "redact", text
        return dec
