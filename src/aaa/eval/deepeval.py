# pyright: reportMissingImports=false, reportAttributeAccessIssue=false, reportGeneralTypeIssues=false
"""DeepEval metrics as AAA metrics, judged through your own LLM gateway.

    from deepeval.metrics import AnswerRelevancyMetric, ToolCorrectnessMetric
    from aaa.eval.deepeval import DeepEvalMetric, DeepEvalJudge

    judge = DeepEvalJudge(my_llm_client, model="judge-model")
    metrics = [DeepEvalMetric(AnswerRelevancyMetric(model=judge, threshold=0.7))]

`DeepEvalJudge` is a `DeepEvalBaseLLM` over an AAA `LLMClient`, so DeepEval's judge calls use your gateway, credentials and budget
controls instead of DeepEval's own OpenAI default. `DeepEvalMetric` builds an `LLMTestCase` from the case result (input, answer,
expected output, tools called, token cost) and calls the metric's async path.

DeepEval is a large dependency that changes quickly (this was written against 4.x; 2.x has different signatures). Install it in an
environment of its own if its pins clash with the rest of your stack."""

from __future__ import annotations

import asyncio
import json
import re
import threading
from collections.abc import Awaitable, Callable
from typing import Any

from aaa.eval.result import CaseResult
from aaa.llm.client import LLMClient
from aaa.llm.types import LLMRequest, Message
from aaa.score import Score


def _run_sync(make: Callable[[], Awaitable[Any]]) -> Any:
    """Run a coroutine to completion from sync code, whether or not a loop is running in this thread."""
    box: dict[str, Any] = {}

    def target() -> None:
        try:
            box["v"] = asyncio.run(make())  # type: ignore[arg-type]
        except BaseException as e:  # noqa: BLE001 - re-raised in the caller's thread
            box["e"] = e

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join()
    if "e" in box:
        raise box["e"]
    return box["v"]


def DeepEvalJudge(client: LLMClient, model: str | None = None, **defaults: Any) -> Any:
    """A `DeepEvalBaseLLM` over `client`. Totals of what it used are on `.calls`, `.tokens` and `.cost`."""
    from deepeval.models import DeepEvalBaseLLM

    class _Judge(DeepEvalBaseLLM):
        def __init__(self) -> None:
            self.calls, self.tokens, self.cost = 0, 0, 0.0
            super().__init__(model or "aaa-judge")

        def load_model(self, *a: Any, **k: Any) -> Any:
            return client

        def get_model_name(self, *a: Any, **k: Any) -> str:
            return model or "aaa-judge"

        def supports_structured_outputs(self) -> bool:
            return True

        async def a_generate(self, prompt: str, schema: Any = None, *a: Any, **k: Any) -> Any:
            resp = await client.complete(
                LLMRequest(
                    model=model,
                    messages=[Message(role="user", content=prompt)],
                    response_schema=schema.model_json_schema() if schema is not None else None,
                    params=dict(defaults),
                )
            )
            self.calls += 1
            self.tokens += resp.usage.total_tokens
            self.cost += resp.cost or 0.0
            content = resp.message.content or ""
            if schema is not None:
                m = re.search(r"\{.*\}", content, re.DOTALL)
                try:
                    return schema.model_validate_json(m.group() if m else content)
                except Exception:  # noqa: BLE001 - DeepEval parses the raw text itself when we cannot
                    return content
            return content

        def generate(self, prompt: str, schema: Any = None, *a: Any, **k: Any) -> Any:
            return _run_sync(lambda: self.a_generate(prompt, schema))

    return _Judge()


class DeepEvalMetric:
    """Run one DeepEval metric (single-turn `LLMTestCase` metrics) on a case result."""

    def __init__(
        self,
        metric: Any,
        *,
        name: str | None = None,
        retrieval_context_key: str = "retrieval_context",
        context_key: str = "context",
    ) -> None:
        self.metric = metric
        self.name: str = name or getattr(metric, "__name__", None) or type(metric).__name__
        self.retrieval_context_key, self.context_key = retrieval_context_key, context_key

    def _test_case(self, r: CaseResult) -> Any:
        from deepeval.test_case import LLMTestCase, ToolCall

        task = r.case.input.get("task")
        expected = r.case.expected
        meta = r.case.metadata
        used = [s for s in r.spans if s.kind == "tool"]
        return LLMTestCase(
            input=task if isinstance(task, str) else json.dumps(r.case.input, default=str),
            actual_output=r.text,
            expected_output=None
            if expected is None
            else expected
            if isinstance(expected, str)
            else json.dumps(expected, default=str),
            retrieval_context=meta.get(self.retrieval_context_key),
            context=meta.get(self.context_key),
            tools_called=[
                ToolCall(
                    name=str(s.attributes.get("tool.name")),
                    input_parameters=s.input if isinstance(s.input, dict) else None,
                    output=s.output,
                )
                for s in used
            ]
            or None,
            expected_tools=[ToolCall(name=n) for n in meta.get("expected_tools", [])] or None,
            token_cost=r.cost,
            completion_time=r.latency_s,
        )

    async def score(self, r: CaseResult) -> Score:
        m = self.metric
        await m.a_measure(self._test_case(r), _show_indicator=False)
        if getattr(m, "error", None):
            return Score(
                name=self.name, value=False, reason=str(m.error)[:300], metadata={"error": True}
            )
        value = float(m.score) if m.score is not None else 0.0
        return Score(
            name=self.name,
            value=value,
            reason=getattr(m, "reason", None),
            metadata={
                "threshold": getattr(m, "threshold", None),
                "success": bool(getattr(m, "success", value >= 0.5)),
            },
        )
