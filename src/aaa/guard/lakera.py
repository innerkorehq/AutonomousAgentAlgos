"""Lakera Guard (now part of Check Point): prompt-injection, PII, content and custom detectors behind one call, `POST https://api.lakera.ai/v2/guard`.

    LakeraGuard(api_key="...", project_id="project-123")

Stage -> message role: `input` as `user`, `output` and `tool_call` as `assistant`, `tool_result` as `tool` (Lakera screens tool messages as untrusted content).
`flagged: true` becomes **block** (category `lakera.<detector>` from the `breakdown`, or `lakera.flagged`). In Lakera's *Detect* mode `flagged` is always false: the guard
then never blocks, by Lakera's design; use the project's enforce mode. The text leaves your process.
Built from Lakera's published request/response description and tested with a mock transport; the per-detector field names in `breakdown` were not confirmed from
the docs, so they are read defensively. Not run against a live account."""

from __future__ import annotations

import os
from collections.abc import Iterable

import httpx

from aaa.guard.types import Action, Finding, GuardContext, GuardResult, Stage

_ROLE = {"input": "user", "output": "assistant", "tool_call": "assistant", "tool_result": "tool"}


class LakeraGuard:
    deterministic = False
    version = "2"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        project_id: str | None = None,
        base_url: str = "https://api.lakera.ai",
        action: Action = "block",
        name: str = "lakera",
        stages: Iterable[Stage] = ("input", "output", "tool_result"),
        timeout_s: float = 10.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("LAKERA_GUARD_API_KEY")
        self.project_id, self.base_url = project_id, base_url.rstrip("/")
        self.action: Action = action
        self.name, self.stages = name, frozenset(stages)
        self._http = httpx.AsyncClient(timeout=timeout_s, transport=transport)

    async def check(self, content: str, ctx: GuardContext, /) -> GuardResult:
        body: dict[str, object] = {
            "messages": [{"role": _ROLE[ctx.stage], "content": content}],
            "breakdown": True,
        }
        if self.project_id:
            body["project_id"] = self.project_id
        r = await self._http.post(
            f"{self.base_url}/v2/guard",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=body,
        )
        r.raise_for_status()
        data = r.json()
        if not data.get("flagged"):
            return GuardResult()
        found = []
        for item in data.get("breakdown") or []:
            if item.get("detected"):
                kind = (
                    item.get("detector_type")
                    or item.get("detector")
                    or item.get("type")
                    or "flagged"
                )
                found.append(Finding(f"lakera.{str(kind).replace('/', '.')}"))
        return GuardResult(self.action, found or [Finding("lakera.flagged")])

    async def aclose(self) -> None:
        await self._http.aclose()
