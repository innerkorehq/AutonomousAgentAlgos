"""OpenAI's moderation endpoint as a guard (`POST /v1/moderations`, free to call; plain HTTP, no SDK needed).

    OpenAIModerationGuard(api_key="sk-...", stages=["input", "output"])
    OpenAIModerationGuard(thresholds={"violence": 0.4, "harassment": 0.2})   # stricter than OpenAI's own flags

The text leaves your process: only use a hosted guard where sending it to the provider is acceptable. A failed call is a guard
failure, handled by the policy's `on_error`. The response is checked against the OpenAI SDK's own types in CI."""

from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Any

import httpx

from aaa.guard.types import Action, Finding, GuardContext, GuardResult, Stage


class OpenAIModerationGuard:
    deterministic = False
    version = "1"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        model: str = "omni-moderation-latest",
        base_url: str = "https://api.openai.com/v1",
        thresholds: dict[str, float] | None = None,
        action: Action = "block",
        name: str = "openai_moderation",
        stages: Iterable[Stage] = ("input", "output"),
        timeout_s: float = 10.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model, self.base_url, self.thresholds = model, base_url.rstrip("/"), thresholds or {}
        self.action: Action = action
        self.name, self.stages = name, frozenset(stages)
        self._http = httpx.AsyncClient(timeout=timeout_s, transport=transport)

    async def check(self, content: str, ctx: GuardContext, /) -> GuardResult:
        r = await self._http.post(
            f"{self.base_url}/moderations",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "input": content},
        )
        r.raise_for_status()
        res: dict[str, Any] = r.json()["results"][0]
        scores: dict[str, float] = res.get("category_scores") or {}
        flagged = {k for k, v in (res.get("categories") or {}).items() if v}
        flagged |= {k for k, limit in self.thresholds.items() if scores.get(k, 0.0) >= limit}
        if (
            self.thresholds
        ):  # explicit thresholds replace OpenAI's own flags for the categories they name
            flagged = {
                k
                for k in flagged
                if k not in self.thresholds or scores.get(k, 0.0) >= self.thresholds[k]
            }
        if not flagged:
            return GuardResult()
        return GuardResult(
            self.action, [Finding(k.replace("/", "."), scores.get(k)) for k in sorted(flagged)]
        )

    async def aclose(self) -> None:
        await self._http.aclose()
