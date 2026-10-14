"""No LLM in the control flow: a deterministic DAG (WorkflowGraph) and a state machine (FSM)."""

import asyncio

from aaa.algos import FSM, Node, WorkflowGraph
from aaa.runtime import Engine
from aaa.tools import ToolHandler, ToolRegistry, tool


@tool
def double(n: int) -> int:
    """x2"""
    return n * 2


@tool
def total(xs: list[int]) -> int:
    """sum"""
    return sum(xs)


async def main() -> None:
    registry = ToolRegistry([double, total])
    graph = WorkflowGraph(
        [
            Node("a", "double", args=lambda results, inp: {"n": inp["n"]}),
            Node(
                "b", "double", args=lambda results, inp: {"n": inp["n"] + 1}
            ),  # runs in parallel with a
            Node(
                "bonus", "double", after=["a"], args={"n": 1000}, when=lambda r, i: r["a"] > 100
            ),  # skipped
            Node(
                "sum",
                "total",
                after=["a", "b", "bonus"],
                args=lambda r, i: {"xs": [r["a"], r["b"]]},
            ),
        ],
        output="sum",
    )

    review = FSM(
        "draft",
        {
            "draft": {"user.submit": "review"},
            "review": {
                "user.reject": "draft",  # cycles are fine in an FSM
                "user.approve": lambda event, inp: (
                    "done" if event.data["by"] in inp["approvers"] else None
                ),
            },
        },
        terminal={"done"},
    )

    async with Engine() as engine:
        ToolHandler(registry).attach(engine)
        print("workflow:", (await engine.run(graph, {"n": 5})).value)

        run = await engine.start(review, {"approvers": ["alice"]})
        await run.signal("user.submit")
        await run.signal("user.reject")
        await run.signal("user.submit")
        await run.signal("user.approve", {"by": "mallory"})  # guard says no: ignored
        await run.signal("user.approve", {"by": "alice"})
        print("fsm path:", " -> ".join((await run.result()).value["path"]))


asyncio.run(main())
