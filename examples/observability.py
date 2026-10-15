"""Observability: turn a run's event log into a trace and send it to a sink. Offline.

Swap `ConsoleSink` for `OTelSink(OTLPSpanExporter(...))`, `LangfuseSink(...)`, `LangSmithSink(...)` or `PhoenixSink(...)`."""

import asyncio
import sys

from aaa.algos import ToolLoop
from aaa.llm import FakeLLMClient, LLMHandler
from aaa.llm.fake import text, tool_call
from aaa.observe import ConsoleSink, InMemorySink, Observer
from aaa.runtime import Engine
from aaa.tools import ToolHandler, ToolRegistry, tool


@tool
def lookup(topic: str) -> str:
    """Look a topic up."""
    return f"{topic}: a fact"


def scripted_model(request):  # stands in for a real LLM
    if request.messages[-1].role == "user":
        return tool_call("lookup", topic="tracing")
    return text("Tracing explained.")


async def main() -> None:
    registry = ToolRegistry([lookup])
    algo = ToolLoop(registry)
    memory_sink = InMemorySink()
    async with Engine() as engine:
        LLMHandler(FakeLLMClient(scripted_model)).attach(engine)
        ToolHandler(registry).attach(engine)
        engine.register(algo)
        # Content (prompts, replies) is NOT exported unless you pass capture_content=True.
        async with Observer(engine, [ConsoleSink(sys.stdout), memory_sink]):
            await engine.run(algo, {"task": "Explain tracing"})
    print("spans exported:", len(memory_sink.spans), "in", memory_sink.batches, "batch")
    print("kinds:", [s.kind for s in memory_sink.spans])


asyncio.run(main())
