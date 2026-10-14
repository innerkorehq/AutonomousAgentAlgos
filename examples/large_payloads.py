"""Big tool outputs stay out of the event log: they are claim-checked into a blob store."""

import asyncio

from aaa.algos import ToolLoop
from aaa.data import InMemoryBlobStore
from aaa.llm import FakeLLMClient, LLMHandler
from aaa.llm.fake import text, tool_call
from aaa.runtime import Engine
from aaa.tools import ToolHandler, ToolRegistry, tool


@tool
def fetch_report() -> str:
    """Returns a large document."""
    return "lorem ipsum " * 5000  # ~60 KB


def model(request):
    last = request.messages[-1]
    return (
        tool_call("fetch_report") if last.role == "user" else text(f"got {len(last.content)} chars")
    )


async def main() -> None:
    registry = ToolRegistry([fetch_report])
    blobs = InMemoryBlobStore()
    async with Engine(blobs=blobs, claim_threshold=2_000) as engine:  # payloads > 2 KB go to blobs
        LLMHandler(FakeLLMClient(model)).attach(engine)
        ToolHandler(registry).attach(engine)
        run = await engine.start(ToolLoop(registry), {"task": "fetch it"})
        outcome = await run.result()
        raw = await engine.store.read(run.run_id)
        stub = next(e for e in raw if e.type == "aaa.tool.completed")
        full = await engine.hydrate(stub)  # handlers and algorithms see this transparently
    print(outcome.value["text"])
    print("stored event keys:", sorted(stub.data), "| blob:", stub.blobs[-1].size, "bytes")
    print("hydrated result length:", len(full.data["result"]))


asyncio.run(main())
