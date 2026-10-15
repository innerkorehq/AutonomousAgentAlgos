# pyright: reportMissingImports=false, reportAttributeAccessIssue=false
"""autoevals (Braintrust) scorers as AAA metrics.

    from autoevals import Factuality, Levenshtein
    from aaa.eval.autoevals import AutoevalsMetric, openai_shim

    metrics = [
        AutoevalsMetric(Levenshtein()),                                           # heuristic: no model
        AutoevalsMetric(Factuality(client=openai_shim(my_llm_client), model="judge-model")),   # judged by YOUR gateway
    ]

autoevals scorers speak the OpenAI SDK. `openai_shim` presents any AAA `LLMClient` as an OpenAI-style client (chat completions with
tool calls, the Responses API autoevals picks for `gpt-5*` model names, and embeddings), so judge calls go through your own gateway
and credentials. autoevals gives no per-score usage, so what the judge spent is the shim's running total (`shim.meter.calls / .tokens / .cost`),
not a per-case number."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from aaa.eval.result import CaseResult
from aaa.llm.client import LLMClient
from aaa.llm.types import LLMRequest, LLMResponse, Message, ToolSpec
from aaa.score import Score


class _Meter:
    def __init__(self) -> None:
        self.calls = 0
        self.tokens = 0
        self.cost = 0.0


def _messages(raw: list[dict[str, Any]]) -> list[Message]:
    out = []
    for m in raw:
        content = m.get("content")
        if isinstance(content, list):  # content parts
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        out.append(
            Message(
                role=m["role"] if m["role"] in ("system", "user", "assistant", "tool") else "user",
                content=content,
            )
        )
    return out


def _tools(raw: list[dict[str, Any]] | None) -> list[ToolSpec]:
    specs = []
    for t in raw or []:
        f = t.get("function", t)
        specs.append(
            ToolSpec(
                name=f["name"],
                description=f.get("description") or "",
                parameters=f.get("parameters") or {"type": "object", "properties": {}},
            )
        )
    return specs


def openai_shim(client: LLMClient, *, model: str | None = None) -> Any:
    """An OpenAI-SDK-shaped object backed by an AAA `LLMClient` (async only: use autoevals' `eval_async`)."""
    meter = _Meter()

    async def call(kwargs: dict[str, Any], messages: list[dict[str, Any]]) -> LLMResponse:
        params = {k: kwargs[k] for k in ("temperature", "max_tokens") if kwargs.get(k) is not None}
        resp = await client.complete(
            LLMRequest(
                model=kwargs.get("model") or model,
                messages=_messages(messages),
                tools=_tools(kwargs.get("tools")),
                params=params,
            )
        )
        meter.calls += 1
        meter.tokens += resp.usage.total_tokens
        meter.cost += resp.cost or 0.0
        return resp

    async def chat_create(**kw: Any) -> dict[str, Any]:
        resp = await call(kw, kw["messages"])
        msg: dict[str, Any] = {"role": "assistant", "content": resp.message.content}
        if resp.message.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
                }
                for c in resp.message.tool_calls
            ]
        return {
            "id": "aaa",
            "object": "chat.completion",
            "created": 0,
            "model": resp.model or kw.get("model") or "",
            "choices": [
                {"index": 0, "finish_reason": resp.finish_reason or "stop", "message": msg}
            ],
            "usage": {
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
                "total_tokens": resp.usage.total_tokens,
            },
        }

    async def responses_create(**kw: Any) -> dict[str, Any]:
        # autoevals converts its chat params to Responses params and the reply back to chat format, so answer in Responses shape
        resp = await call(kw, kw["input"])
        output: list[dict[str, Any]] = []
        if resp.message.content:
            output.append({"type": "output_text", "text": resp.message.content})
        for c in resp.message.tool_calls:
            output.append(
                {
                    "type": "function_call",
                    "call_id": c.id,
                    "name": c.name,
                    "arguments": json.dumps(c.arguments),
                }
            )
        return {"id": "aaa", "model": resp.model or "", "output": output}

    async def embeddings_create(**kw: Any) -> dict[str, Any]:
        inp = kw["input"]
        texts = [inp] if isinstance(inp, str) else list(inp)
        res = await client.embed(texts, kw.get("model"))
        return {
            "object": "list",
            "model": kw.get("model") or "",
            "data": [
                {"object": "embedding", "index": i, "embedding": v}
                for i, v in enumerate(res.vectors)
            ],
            "usage": {"prompt_tokens": 0, "total_tokens": 0},
        }

    async def unsupported(**kw: Any) -> Any:
        raise NotImplementedError("moderations are not available through the AAA shim")

    shim = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=chat_create)),
        responses=SimpleNamespace(create=responses_create),
        embeddings=SimpleNamespace(create=embeddings_create),
        moderations=SimpleNamespace(create=unsupported),
    )
    shim.meter = meter  # type: ignore[attr-defined]
    return shim


class AutoevalsMetric:
    """Run one autoevals scorer on the case: `output` is the agent's answer, `expected` the case's, `input` the task text."""

    def __init__(
        self, scorer: Any, *, name: str | None = None, pass_threshold: float | None = None
    ) -> None:
        self.scorer, self.pass_threshold = scorer, pass_threshold
        n = getattr(scorer, "_name", None)  # a method on autoevals scorers
        self.name: str = str(name or (n() if callable(n) else n) or type(scorer).__name__.lower())

    async def score(self, r: CaseResult) -> Score:
        task = r.case.input.get("task", r.case.input)
        out = await self.scorer.eval_async(
            output=r.text,
            expected=r.case.expected,
            input=task if isinstance(task, str) else json.dumps(task, default=str),
        )
        if out.error is not None:
            return Score(
                name=self.name,
                value=False,
                reason=f"{type(out.error).__name__}: {out.error}"[:300],
                metadata={"error": True},
            )
        if out.score is None:  # autoevals: "not applicable"
            return Score(
                name=self.name,
                value=True,
                reason="not applicable",
                metadata=dict(out.metadata or {}),
            )
        value: float | bool = float(out.score)
        reason = (out.metadata or {}).get("rationale")
        s = Score(
            name=self.name,
            value=value,
            reason=str(reason) if reason else None,
            metadata=dict(out.metadata or {}),
        )
        if self.pass_threshold is not None:
            s.metadata["passed"] = value >= self.pass_threshold
        return s
