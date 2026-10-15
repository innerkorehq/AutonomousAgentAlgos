# pyright: reportMissingImports=false, reportAttributeAccessIssue=false
"""AWS Bedrock Guardrails through `ApplyGuardrail`, which checks text WITHOUT calling any model (so it guards any model, not only Bedrock's).

    BedrockGuard("gr-abc123", "1", region_name="us-east-1")            # boto3 default credentials
    BedrockGuard("gr-abc123", "DRAFT", client=my_bedrock_runtime_client)

Stage -> `source`: user `input` and untrusted `tool_result` are sent as `INPUT`; the model's `output` and its `tool_call` arguments as `OUTPUT`.
`GUARDRAIL_INTERVENED` becomes **block**, except when every policy that fired only masked (`ANONYMIZED` PII) and Bedrock returned replacement text: then it
is **redact** with that text. Findings are `content.<type>` (HATE, PROMPT_ATTACK, ...), `topic.<name>`, `word.custom` / `word.<managed>`, `pii.<type>`,
`regex.<name>`, `grounding.<type>`; the matched text is never copied into a finding. The boto3 client is synchronous and runs in a worker thread.
The request and response are validated against botocore's service model in the tests; not run against a live AWS account."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from typing import Any

from aaa.guard.types import Finding, GuardContext, GuardResult, Stage

_SOURCE = {"input": "INPUT", "tool_result": "INPUT", "output": "OUTPUT", "tool_call": "OUTPUT"}


def _findings(assessments: list[dict[str, Any]]) -> tuple[list[Finding], bool, bool]:
    """(findings, any_blocked, any_masked)"""
    out: list[Finding] = []
    blocked = masked = False

    def note(category: str, action: str | None, score: str | None = None) -> None:
        nonlocal blocked, masked
        if action == "ANONYMIZED":
            masked = True
        elif action not in (None, "NONE"):
            blocked = True
        out.append(Finding(category, detail=score))

    for a in assessments:
        for f in (a.get("contentPolicy") or {}).get("filters", []):
            if f.get("detected", True):
                note(f"content.{f['type'].lower()}", f.get("action"), f.get("confidence"))
        for t in (a.get("topicPolicy") or {}).get("topics", []):
            if t.get("detected", True):
                note(f"topic.{t['name']}", t.get("action"))
        w = a.get("wordPolicy") or {}
        for cw in w.get("customWords", []):
            if cw.get("detected", True):
                note("word.custom", cw.get("action"))
        for mw in w.get("managedWordLists", []):
            if mw.get("detected", True):
                note(f"word.{mw['type'].lower()}", mw.get("action"))
        s = a.get("sensitiveInformationPolicy") or {}
        for p in s.get("piiEntities", []):
            if p.get("detected", True):
                note(f"pii.{p['type'].lower()}", p.get("action"))
        for r in s.get("regexes", []):
            if r.get("detected", True):
                note(f"regex.{r['name']}", r.get("action"))
        for g in (a.get("contextualGroundingPolicy") or {}).get("filters", []):
            if g.get("detected", True):
                note(f"grounding.{g['type'].lower()}", g.get("action"), str(g.get("score")))
    return out, blocked, masked


class BedrockGuard:
    deterministic = False
    version = "1"

    def __init__(
        self,
        guardrail_id: str,
        guardrail_version: str = "DRAFT",
        *,
        client: Any = None,
        name: str = "bedrock",
        stages: Iterable[Stage] = ("input", "output", "tool_result"),
        **client_kwargs: Any,
    ) -> None:
        self.guardrail_id, self.guardrail_version = guardrail_id, guardrail_version
        self.name, self.stages = name, frozenset(stages)
        self._client = client
        self._kw = client_kwargs
        self.version = f"1:{guardrail_id}:{guardrail_version}"

    def _c(self) -> Any:
        if self._client is None:
            import boto3

            self._client = boto3.client("bedrock-runtime", **self._kw)
        return self._client

    async def check(self, content: str, ctx: GuardContext, /) -> GuardResult:
        resp = await asyncio.to_thread(
            self._c().apply_guardrail,
            guardrailIdentifier=self.guardrail_id,
            guardrailVersion=self.guardrail_version,
            source=_SOURCE[ctx.stage],
            content=[{"text": {"text": content}}],
        )
        if resp.get("action") != "GUARDRAIL_INTERVENED":
            return GuardResult()
        findings, blocked, masked = _findings(resp.get("assessments") or [])
        texts = [o["text"] for o in resp.get("outputs") or [] if o.get("text") is not None]
        if masked and not blocked and texts:
            return GuardResult("redact", findings, "".join(texts))
        return GuardResult("block", findings or [Finding("bedrock.intervened")])
