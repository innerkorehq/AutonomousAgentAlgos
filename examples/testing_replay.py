"""Test agents deterministically: record a run once, replay its LLM answers forever, no network."""

import asyncio

from aaa.algos import ToolLoop
from aaa.llm import FakeLLMClient, LLMHandler, ReplayLLMClient
from aaa.llm.fake import text, tool_call
from aaa.runtime import Engine
from aaa.tools import ToolHandler, ToolRegistry, tool


@tool
def add(a: int, b: int) -> int:
    """Add."""
    return a + b


async def run_agent(llm):
    registry = ToolRegistry([add])
    async with Engine() as engine:
        LLMHandler(llm).attach(engine)
        ToolHandler(registry).attach(engine)
        run = await engine.start(ToolLoop(registry), {"task": "2 + 3?"})
        outcome = await run.result()
        return outcome, await engine.store.read(run.run_id)


async def main() -> None:
    # 1) "record": in real life this is your live model; its answers are saved in the event log
    first = FakeLLMClient(
        lambda r: tool_call("add", a=2, b=3) if r.messages[-1].role == "user" else text("5")
    )
    outcome1, log = await run_agent(first)

    # 2) "replay": serve the recorded llm.completed events back, in order. No scripting, no network.
    outcome2, _ = await run_agent(ReplayLLMClient(log))
    assert outcome1.value == outcome2.value
    print("replayed identically:", outcome2.value["text"])


asyncio.run(main())
