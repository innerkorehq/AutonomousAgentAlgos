from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

from aaa.algos.base import finish, llm_request
from aaa.algos.loops import ToolLoop
from aaa.core.outcome import OutcomeKind
from aaa.data.index import Index
from aaa.events import Event, Projection
from aaa.llm.types import LLMRequest, LLMResponse, Message
from aaa.runtime import Algorithm, Ctx
from aaa.tools import Tool, ToolContext, ToolError, ToolRegistry, search_tool

_ANS = re.compile(r"Answer:\s*(.+)", re.IGNORECASE)


def default_extract(text: str) -> str:
    m = _ANS.findall(text)
    return (m[-1] if m else (text.strip().splitlines() or [""])[-1]).strip()


def _norm(s: str) -> str:
    return re.sub(r"[^\w\s.-]", "", s.lower()).strip().rstrip(".")


class _Samples(BaseModel):
    texts: list[str] = Field(
        default_factory=list
    )  # raw model text; `extract` is applied at read time
    failures: int = 0


def _apply_sample(st: _Samples, e: Event) -> _Samples:
    if e.type == "aaa.llm.completed":
        st.texts.append(LLMResponse.model_validate(e.data["response"]).message.content or "")
    elif e.type == "aaa.llm.failed":
        st.failures += 1
    return st


SAMPLES = Projection(_Samples, _apply_sample, name="aaa.samples", version=1, state_type=_Samples)


class SelfConsistency:
    """Sample N independent answers in parallel and return the majority (ties -> earliest).
    Only worthwhile when answers can be compared exactly (math, classification, extraction)."""

    name = "self_consistency"

    def __init__(
        self,
        n: int = 5,
        *,
        temperature: float = 0.8,
        model: str | None = None,
        system: str = "Think step by step, then end with a line 'Answer: <final answer>'.",
        extract: Callable[[str], str] = default_extract,
    ) -> None:
        self.n, self.temperature, self.model, self.system, self.extract = (
            n,
            temperature,
            model,
            system,
            extract,
        )

    def subscriptions(self) -> list[str]:
        return ["aaa.run.started", "aaa.llm.completed", "aaa.llm.failed"]

    async def on_event(self, e: Event, ctx: Ctx) -> list[Event]:
        if e.type == "aaa.run.started":
            task = e.data["input"].get("task", "")
            req = LLMRequest(
                model=self.model,
                params={"temperature": self.temperature},
                messages=[
                    Message(role="system", content=self.system),
                    Message(role="user", content=str(task)),
                ],
            )
            return [llm_request(e, req, f"s{i}") for i in range(self.n)]
        st = await ctx.project(SAMPLES)
        samples = [self.extract(t) for t in st.texts]
        failures = st.failures
        if len(samples) + failures < self.n:
            return []
        if not samples:
            return [finish(e, OutcomeKind.FAILED, reason="all samples failed")]
        votes = Counter(_norm(s) for s in samples)
        best, count = max(
            votes.items(), key=lambda kv: (kv[1], -[_norm(s) for s in samples].index(kv[0]))
        )
        answer = next(s for s in samples if _norm(s) == best)
        return [
            finish(
                e,
                OutcomeKind.SUCCESS,
                {
                    "answer": answer,
                    "votes": count,
                    "samples": len(samples),
                    "agreement": round(count / len(samples), 3),
                },
            )
        ]


class RAGAgent(ToolLoop):
    """Agentic RAG: a ToolLoop whose tool is retrieval over any `Index`. The model decides
    when/what to retrieve and is told to ground and cite. Attach `ToolHandler(rag.tools)`."""

    name = "rag_agent"
    _PROMPT = (
        "Answer using only evidence retrieved with the search tool. Cite passage ids like [d1]. "
        "If the evidence is insufficient, say so instead of guessing."
    )

    def __init__(self, index: Index, *, k: int = 5, **kw: Any) -> None:
        kw.setdefault("system", self._PROMPT)
        super().__init__(ToolRegistry([search_tool(index, k=k)]), **kw)


class Supervisor(ToolLoop):
    """Centralized coordinator: an LLM delegates sub-tasks to named worker algorithms (each a
    child run: own budget accounting, cancelled with the parent). Parallelism is bounded by the
    'delegate' pool. Attach `ToolHandler(supervisor.tools)`."""

    name = "supervisor"

    def __init__(
        self,
        workers: dict[str, tuple[Algorithm, str]],
        *,
        extra_tools: list[Tool] | None = None,
        **kw: Any,
    ) -> None:
        reg = ToolRegistry(extra_tools or [])
        for wname, (algo, desc) in workers.items():
            reg.add(self._delegate_tool(wname, algo, desc))
        roster = "\n".join(f"- delegate_{k}: {d}" for k, (_, d) in workers.items())
        kw.setdefault(
            "system",
            f"You coordinate specialist workers. Delegate sub-tasks (possibly in parallel), then combine their results.\n{roster}",
        )
        super().__init__(reg, **kw)

    @staticmethod
    def _delegate_tool(wname: str, algo: Algorithm, desc: str) -> Tool:
        async def delegate(task: str, ctx: ToolContext) -> Any:
            handle = await ctx.engine.start(
                algo, {"task": task}, parent=ctx.run_id, tag=ctx.call_id
            )
            out = await handle.result()
            if not out.ok:
                raise ToolError(f"worker {wname} ended {out.kind.value}: {out.reason}")
            return (
                out.value["text"]
                if isinstance(out.value, dict) and "text" in out.value
                else out.value
            )

        return Tool(
            delegate,
            name=f"delegate_{wname}",
            description=f"Delegate a self-contained sub-task. {desc}",
            pool="delegate",
            timeout_s=600,
        )
