# pyright: reportMissingImports=false, reportAttributeAccessIssue=false, reportGeneralTypeIssues=false
"""Ragas metrics (the `ragas.metrics.collections` API, ragas 0.4) as AAA metrics, judged through your own gateway.

    from ragas.metrics.collections import Faithfulness, ContextPrecision, ExactMatch
    from aaa.eval.ragas import RagasMetric, RagasJudge

    judge = RagasJudge(my_llm_client, model="judge-model")
    metrics = [RagasMetric(Faithfulness(llm=judge)), RagasMetric(ExactMatch())]

A case supplies what the metric asks for by parameter name: `user_input` (the task), `response` (the answer), `reference` (the case's
`expected`) and `retrieved_contexts` (`case.metadata["retrieval_context"]`, a list of strings). A metric asking for something else is
rejected with a clear message rather than called with guesses.

Ragas pulls in LangChain, and a current ragas (0.4.3) fails to import against the newest langchain-community / langchain-core
(`ModuleNotFoundError: langchain_community.chat_models.vertexai`, then `ContextOverflowError`). Pin `langchain-community<0.4`,
`langchain-core<1` and `langchain-openai<1` until ragas catches up, and install it in an environment of its own."""

from __future__ import annotations

import asyncio
import inspect
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


def RagasJudge(client: LLMClient, model: str | None = None, **params: Any) -> Any:
    """An `InstructorBaseRagasLLM` over `client`: structured replies are requested with the pydantic model's JSON Schema."""
    from ragas.llms.base import InstructorBaseRagasLLM

    class _Judge(InstructorBaseRagasLLM):
        def __init__(self) -> None:
            self.calls, self.tokens, self.cost = 0, 0, 0.0

        async def agenerate(self, prompt: str, response_model: Any) -> Any:
            resp = await client.complete(
                LLMRequest(
                    model=model,
                    messages=[Message(role="user", content=prompt)],
                    response_schema=response_model.model_json_schema(),
                    params=dict(params),
                )
            )
            self.calls += 1
            self.tokens += resp.usage.total_tokens
            self.cost += resp.cost or 0.0
            content = resp.message.content or ""
            m = re.search(r"\{.*\}", content, re.DOTALL)
            return response_model.model_validate(json.loads(m.group() if m else content))

        def generate(self, prompt: str, response_model: Any) -> Any:
            return _run_sync(lambda: self.agenerate(prompt, response_model))

    return _Judge()


class RagasMetric:
    def __init__(self, metric: Any, *, name: str | None = None) -> None:
        self.metric = metric
        self.name: str = name or getattr(metric, "name", None) or type(metric).__name__
        self._params = [p for p in inspect.signature(metric.ascore).parameters if p != "self"]

    def _kwargs(self, r: CaseResult) -> dict[str, Any]:
        task = r.case.input.get("task")
        have: dict[str, Any] = {
            "user_input": task if isinstance(task, str) else json.dumps(r.case.input, default=str),
            "response": r.text,
            "reference": r.case.expected
            if isinstance(r.case.expected, str) or r.case.expected is None
            else json.dumps(r.case.expected, default=str),
            "retrieved_contexts": r.case.metadata.get("retrieval_context"),
        }
        unknown = [p for p in self._params if p not in have]
        if unknown:
            raise ValueError(
                f"{self.name}: cannot supply {unknown} from an eval case (supported: {sorted(have)})"
            )
        missing = [p for p in self._params if have[p] is None]
        if missing:
            raise ValueError(f"{self.name}: case {r.case.id!r} has no {missing}")
        return {p: have[p] for p in self._params}

    async def score(self, r: CaseResult) -> Score:
        res = await self.metric.ascore(**self._kwargs(r))
        v = res.value
        if v is None:
            return Score(
                name=self.name,
                value=False,
                reason=getattr(res, "reason", None) or "no value",
                metadata={"error": True},
            )
        return Score(name=self.name, value=float(v), reason=getattr(res, "reason", None))
