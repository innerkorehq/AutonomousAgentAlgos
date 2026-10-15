"""Azure AI Content Safety **Prompt Shields**: detects user-prompt attacks (jailbreaks) and document attacks (instructions hidden in content the model reads).

    AzurePromptShieldGuard(endpoint="https://<resource>.cognitiveservices.azure.com", api_key="...")

`input` text is sent as `userPrompt`; `tool_result` (and `output`) text as a `documents` entry, which is the case Prompt Shields calls *document attacks*: injected
instructions in content, not harmful content in general. `attackDetected` becomes **block** (category `prompt_attack.user` / `prompt_attack.document`).
Plain HTTP (`POST <endpoint>/contentsafety/text:shieldPrompt?api-version=2024-09-01`, header `Ocp-Apim-Subscription-Key`); the text leaves your process.
Built from Microsoft's published request/response shapes and tested with a mock transport; not run against a live resource."""

from __future__ import annotations

import os
from collections.abc import Iterable

import httpx

from aaa.guard.types import Action, Finding, GuardContext, GuardResult, Stage

API_VERSION = "2024-09-01"


class AzurePromptShieldGuard:
    deterministic = False
    version = API_VERSION

    def __init__(
        self,
        endpoint: str | None = None,
        api_key: str | None = None,
        *,
        action: Action = "block",
        name: str = "azure_prompt_shield",
        stages: Iterable[Stage] = ("input", "tool_result"),
        timeout_s: float = 10.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.endpoint = (endpoint or os.environ.get("CONTENT_SAFETY_ENDPOINT", "")).rstrip("/")
        self.api_key = api_key or os.environ.get("CONTENT_SAFETY_KEY")
        self.action: Action = action
        self.name, self.stages = name, frozenset(stages)
        self._http = httpx.AsyncClient(timeout=timeout_s, transport=transport)

    async def check(self, content: str, ctx: GuardContext, /) -> GuardResult:
        as_prompt = ctx.stage == "input"
        body = {
            "userPrompt": content if as_prompt else "",
            "documents": [] if as_prompt else [content],
        }
        r = await self._http.post(
            f"{self.endpoint}/contentsafety/text:shieldPrompt",
            params={"api-version": API_VERSION},
            headers={"Ocp-Apim-Subscription-Key": self.api_key or ""},
            json=body,
        )
        r.raise_for_status()
        data = r.json()
        found = []
        if (data.get("userPromptAnalysis") or {}).get("attackDetected"):
            found.append(Finding("prompt_attack.user"))
        if any(d.get("attackDetected") for d in data.get("documentsAnalysis") or []):
            found.append(Finding("prompt_attack.document"))
        return GuardResult(self.action, found) if found else GuardResult()

    async def aclose(self) -> None:
        await self._http.aclose()
