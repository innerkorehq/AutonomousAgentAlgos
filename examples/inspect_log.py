"""A run is just a list of events. Print the log of a finished run."""

import asyncio

from aaa.algos import ToolLoop
from aaa.llm import FakeLLMClient, LLMHandler
from aaa.llm.fake import text, tool_call
from aaa.runtime import Engine
from aaa.tools import ToolHandler, ToolRegistry, tool


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


def scripted_model(request):
    last = request.messages[-1]
    return tool_call("multiply", a=17, b=23) if last.role == "user" else text(f"= {last.content}")


async def main() -> None:
    registry = ToolRegistry([multiply])
    async with Engine() as engine:
        LLMHandler(FakeLLMClient(scripted_model)).attach(engine)
        ToolHandler(registry).attach(engine)
        run = await engine.start(ToolLoop(registry), {"task": "What is 17 times 23?"})
        outcome = await run.result()
        for event in await engine.store.read(run.run_id):
            print(event.seq, event.type)
    print("outcome:", outcome.kind.value)


asyncio.run(main())
