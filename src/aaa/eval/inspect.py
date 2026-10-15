# pyright: reportMissingImports=false, reportAttributeAccessIssue=false, reportGeneralTypeIssues=false
"""Use an AAA agent as the *solver* of an Inspect AI task.

Inspect owns the evaluation loop (datasets, epochs, concurrency, sandboxes, scorers, logs and its viewer); AAA supplies the agent
being evaluated:

    from inspect_ai import Task, eval
    from inspect_ai.dataset import Sample
    from inspect_ai.scorer import includes
    from aaa.eval.inspect import aaa_solver

    @asynccontextmanager
    async def agent():                      # a fresh engine for every sample
        async with Engine() as engine:
            ...attach handlers, register algo...
            yield engine, algo

    eval(Task(dataset=[Sample(input="add 2 and 3", target="5")], solver=aaa_solver(agent), scorer=includes()), model="mockllm/model")

Use this when you want Inspect's scorers, sandboxes and logs; use `aaa.eval.EvalSuite` when you want AAA's trajectory metrics,
budgets and CI gate. The `model` Inspect is given is not used by the agent (the agent calls its own `LLMClient`); model-graded Inspect
scorers use Inspect's own model."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from typing import Any

from aaa.eval.result import default_extract
from aaa.runtime import Budget, Engine
from aaa.runtime.engine import Algorithm


def aaa_solver(
    setup: Callable[[], AbstractAsyncContextManager[tuple[Engine, Algorithm]]],
    *,
    make_input: Callable[[str], dict[str, Any]] = lambda text: {"task": text},
    extract: Callable[[Any], str] = default_extract,
    budget: Budget | None = None,
) -> Any:
    """An Inspect `Solver` that runs one AAA run per sample and puts its answer in `state.output`."""
    from inspect_ai.model import ChatMessageAssistant, ModelOutput
    from inspect_ai.solver import Generate, TaskState, solver

    @solver
    def _aaa_solver() -> Any:
        async def solve(state: TaskState, generate: Generate) -> TaskState:
            async with setup() as (engine, algo):
                outcome = await engine.run(
                    algo, make_input(state.input_text), budget=budget, meta={"aaa_eval": True}
                )
            answer = extract(outcome.value) if outcome.ok else ""
            if not outcome.ok:
                state.metadata["aaa_outcome"] = f"{outcome.kind.value}: {outcome.reason}"
            state.output = ModelOutput.from_content("aaa", answer)
            state.messages.append(ChatMessageAssistant(content=answer))
            state.completed = True
            return state

        return solve

    return _aaa_solver()
