"""Evals: run a dataset through an agent, score it, gate on the result. Offline (a scripted model stands in)."""

import asyncio

from aaa.algos import ToolLoop
from aaa.eval import Budgeted, Contains, Dataset, EvalSuite, ToolCalled
from aaa.llm import FakeLLMClient, LLMHandler
from aaa.llm.fake import text, tool_call
from aaa.runtime import Engine
from aaa.tools import ToolHandler, ToolRegistry, tool


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def scripted_model(request):  # stands in for a real LLM
    last = request.messages[-1]
    if last.role == "user":
        nums = [int(w) for w in (last.content or "").split() if w.isdigit()]
        if len(nums) < 2:
            return text("I do not know.")  # a bad agent on some inputs
        return tool_call("add", a=nums[0], b=nums[1])
    return text(f"The answer is {last.content}")


DATASET = Dataset.from_list(
    [
        {"id": "small", "input": "add 2 3", "expected": "5"},
        {"id": "large", "input": "add 40 2", "expected": "42"},
        {"id": "vague", "input": "add the numbers", "expected": "7"},
    ]
)


async def main() -> None:
    registry = ToolRegistry([add])
    algo = ToolLoop(registry)
    async with Engine() as engine:
        LLMHandler(FakeLLMClient(scripted_model)).attach(engine)
        ToolHandler(registry).attach(engine)
        engine.register(algo)
        suite = EvalSuite(DATASET, algo, [Contains(), ToolCalled("add"), Budgeted(steps=5)])
        report = await suite.run(engine)
    print(f"pass rate {report.pass_rate:.0%} over {len(report.results)} runs")
    for name, s in report.summary().items():
        print(f"  {name}: mean {s.mean:.2f}")
    print("failed:", [r.case.id for r in report.failures])
    try:
        report.assert_ok(min_pass_rate=0.9)  # in CI this fails the build
    except AssertionError as e:
        print("gate:", str(e).splitlines()[1])


asyncio.run(main())
