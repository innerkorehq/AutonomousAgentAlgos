"""Context management: a 25-step run inside a deliberately small window. Runs offline."""

import asyncio

from aaa.algos import ToolLoop
from aaa.context import ContextPolicy, HeuristicCounter
from aaa.llm import FakeLLMClient, LLMHandler
from aaa.llm.fake import text, tool_call
from aaa.runtime import Engine
from aaa.tools import ToolHandler, ToolRegistry, tool

WINDOW = 4_000  # tokens the model can take as input (use your model's real limit)
STEPS = 25


@tool
def lookup(topic: str) -> str:
    """Look a topic up (returns a long document)."""
    return f"Notes on {topic}: " + "lorem ipsum " * 150


def run_agent(request, sizes):  # stands in for the real LLM
    sizes.append(HeuristicCounter().count(request.messages, request.tools))
    if len(sizes) <= STEPS:
        return tool_call("lookup", topic=f"topic {len(sizes)}")
    return text("Done: I looked up all 25 topics.")


async def run(policy):
    sizes: list[int] = []
    async with Engine() as engine:
        LLMHandler(FakeLLMClient(lambda r: run_agent(r, sizes)), context=policy).attach(engine)
        registry = ToolRegistry([lookup])
        ToolHandler(registry).attach(engine)
        handle = await engine.start(
            ToolLoop(registry, max_turns=40), {"task": "research everything"}
        )
        outcome = await handle.result()
        log = await engine.store.read(handle.run_id)
    return outcome, sizes, log


async def main() -> None:
    _, unmanaged, _ = await run(None)
    summarizer = FakeLLMClient(
        lambda r: text("Looked up the earlier topics; each returned long notes.")
    )
    policy = ContextPolicy(max_input_tokens=WINDOW, reserve_output=0, summarizer=summarizer)
    outcome, managed, log = await run(policy)

    print(f"without a policy the largest request was {max(unmanaged)} tokens (window {WINDOW})")
    print(f"with a policy the largest request was {max(managed)} tokens")
    print(f"the model was called {len(managed)} times and finished: {outcome.value['text']}")
    reports = [
        e.data["context"] for e in log if e.type == "aaa.llm.completed" and "context" in e.data
    ]
    used = sorted({a.split("(")[0] for r in reports for a in r["applied"]})
    print(f"compaction was applied on {len(reports)} calls, using: {used}")
    print(f"summaries written to the log: {sum(1 for e in log if e.type == 'aaa.context.summary')}")
    requests = [e for e in log if e.type == "aaa.llm.requested"]
    print(
        f"the log still holds every message: {all(len(e.data['conv']['delta']) <= 2 for e in requests[1:])}"
    )


asyncio.run(main())
