# pyright: reportMissingImports=false, reportAttributeAccessIssue=false
"""NVIDIA NeMo Guardrails: your configured input/output rails (Colang flows, actions, self-check prompts) as a guard.

    from nemoguardrails import LLMRails, RailsConfig
    from aaa.guard.nemo import NeMoGuard

    rails = LLMRails(RailsConfig.from_path("./guardrails_config"))
    NeMoGuard(rails)

Uses `LLMRails.check_async(messages, rail_types=...)` (NeMo Guardrails 0.24): `input` and `tool_result` run the **input** rails (the text is presented as a user message),
`output` and `tool_call` run the **output** rails. `blocked` -> **block** (category `nemo.<rail>`), `modified` -> **redact** with NeMo's rewritten content, `passed` -> allow.
NeMo only exposes input and output rails, so the dialog, retrieval and execution rails are not used. Output rails that look at the user's message see a placeholder, because a
guard only receives the text being checked. Rails that call an LLM spend tokens that AAA does not see. Verified with the real library and a configuration whose rail is a
Python action (no model needed); rails that need an LLM were not run."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from aaa.guard.types import Finding, GuardContext, GuardResult, Stage

_INPUT_STAGES = ("input", "tool_result")


class NeMoGuard:
    deterministic = False

    def __init__(
        self,
        rails: Any,
        *,
        name: str = "nemo",
        stages: Iterable[Stage] = ("input", "output", "tool_result"),
    ) -> None:
        self.rails, self.name, self.stages = rails, name, frozenset(stages)
        try:
            from importlib.metadata import version

            self.version = f"1:{version('nemoguardrails')}"
        except Exception:  # noqa: BLE001 - the version is only a cache key
            self.version = "1"

    async def check(self, content: str, ctx: GuardContext, /) -> GuardResult:
        from nemoguardrails.rails.llm.options import RailStatus, RailType

        if ctx.stage in _INPUT_STAGES:
            messages = [{"role": "user", "content": content}]
            types = [RailType.INPUT]
        else:
            messages = [
                {"role": "user", "content": "(the user's message is not available to a guard)"},
                {"role": "assistant", "content": content},
            ]
            types = [RailType.OUTPUT]
        res = await self.rails.check_async(messages, rail_types=types)
        if res.status == RailStatus.PASSED:
            return GuardResult()
        rail = re.sub(r"\W+", "_", str(res.rail or "rail")).strip("_").lower()
        if res.status == RailStatus.MODIFIED and isinstance(res.content, str):
            return GuardResult("redact", [Finding(f"nemo.{rail}")], res.content)
        return GuardResult("block", [Finding(f"nemo.{rail}")])
