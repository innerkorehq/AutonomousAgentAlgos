"""Quickstart: a tool-using agent in ~25 lines. Runs offline with a scripted model."""

import asyncio

from aaa.algos import ToolLoop
from aaa.llm import FakeLLMClient, LLMHandler
from aaa.llm.fake import text, tool_call
from aaa.runtime import Budget, Engine
from aaa.tools import ToolHandler, ToolRegistry, tool


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


def scripted_model(request):  # stands in for a real LLM (see "LLM gateways" to swap it)
    last = request.messages[-1]
    if last.role == "user":  # first turn: the model asks for the tool
        return tool_call("multiply", a=17, b=23)
    return text(f"17 x 23 = {last.content}")  # second turn: it answers using the tool result


async def main() -> None:
    registry = ToolRegistry([multiply])
    async with Engine() as engine:
        LLMHandler(FakeLLMClient(scripted_model)).attach(engine)  # who answers LLM requests
        ToolHandler(registry).attach(engine)  # who runs tool calls
        outcome = await engine.run(
            ToolLoop(registry),
            {"task": "What is 17 times 23?"},
            budget=Budget(max_steps=20, deadline_s=30),
        )
    print(outcome.kind.value, "-", outcome.value["text"])


asyncio.run(main())
