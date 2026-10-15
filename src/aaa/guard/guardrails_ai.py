# pyright: reportMissingImports=false, reportAttributeAccessIssue=false
"""Guardrails AI validators (the Hub, or your own) as a guard.

    from guardrails import AsyncGuard
    from guardrails.hub import ToxicLanguage            # after `guardrails hub install hub://guardrails/toxic_language`
    from aaa.guard.guardrails_ai import GuardrailsAIGuard

    GuardrailsAIGuard(AsyncGuard().use(ToxicLanguage(on_fail="noop")))

Mapping, from the `ValidationOutcome`: validation failed -> **block** (or the `action` you choose); validation passed but the output changed (a validator with
`on_fail="fix"`) -> **redact** with the fixed text; passed unchanged -> allow. A validator configured `on_fail="exception"` is caught and treated as a block. Findings
are `guardrails.<validator>`. Use `on_fail="noop"` or `"fix"` on validators: AAA, not the guard, decides what a failure means.
Guardrails AI sends usage telemetry over OTLP unless disabled (set `OTEL_SDK_DISABLED=true`); that is the library's behaviour, not AAA's.
Sync `Guard`s run in a worker thread. Verified with the real library (0.11) and a custom validator; Hub validators are not installed in CI."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Iterable
from typing import Any

from aaa.guard.types import Action, Finding, GuardContext, GuardResult, Stage


class GuardrailsAIGuard:
    deterministic = False

    def __init__(
        self,
        guard: Any,
        *,
        action: Action = "block",
        name: str = "guardrails_ai",
        stages: Iterable[Stage] = ("input", "output", "tool_result"),
    ) -> None:
        self.guard, self.name, self.stages = guard, name, frozenset(stages)
        self.action: Action = action
        try:
            from importlib.metadata import version

            self.version = f"1:{version('guardrails-ai')}"
        except Exception:  # noqa: BLE001 - the version is only a cache key
            self.version = "1"

    async def _validate(self, content: str) -> Any:
        fn = self.guard.validate
        if inspect.iscoroutinefunction(fn) or inspect.iscoroutinefunction(
            getattr(fn, "__wrapped__", None)
        ):
            return await fn(content)
        out = await asyncio.to_thread(fn, content)
        return await out if inspect.isawaitable(out) else out

    async def check(self, content: str, ctx: GuardContext, /) -> GuardResult:
        try:
            from guardrails.errors import ValidationError
        except ImportError:  # pragma: no cover - guardrails-ai is required to get here at all
            ValidationError = ()  # type: ignore[assignment]
        try:
            out = await self._validate(content)
        except ValidationError:
            return GuardResult(self.action, [Finding("guardrails.validation_error")])
        failed = [
            s
            for s in (out.validation_summaries or [])
            if str(getattr(s, "validator_status", "")).lower() == "fail"
        ]
        findings = [
            Finding(f"guardrails.{s.validator_name}", detail=getattr(s, "failure_reason", None))
            for s in failed
        ]
        if not out.validation_passed:
            return GuardResult(self.action, findings or [Finding("guardrails.failed")])
        fixed = out.validated_output
        if isinstance(fixed, str) and fixed != content:
            return GuardResult("redact", findings or [Finding("guardrails.fixed")], fixed)
        return GuardResult()
