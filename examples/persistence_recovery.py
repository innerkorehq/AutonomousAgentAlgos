"""Durable runs: kill the process mid-run, start a new one, and the run finishes where it left off."""

import asyncio
import tempfile
from pathlib import Path

from aaa.algos import ToolLoop
from aaa.events import SQLiteEventStore
from aaa.llm import FakeLLMClient, LLMHandler
from aaa.llm.fake import text, tool_call
from aaa.runtime import Engine
from aaa.tools import ToolHandler, ToolRegistry, tool


@tool
def lookup(key: str) -> str:
    """Look something up."""
    return f"value-of-{key}"


class HangsOnSecondCall:
    """A model whose second call never returns, like a process that dies mid-request."""

    calls = 0

    async def complete(self, request):
        self.calls += 1
        if self.calls == 1:
            return tool_call("lookup", key="x")
        await asyncio.Event().wait()

    async def embed(self, *a, **k): ...
    async def aclose(self): ...


async def boot(db: Path, llm):
    store = SQLiteEventStore(db)
    engine = await Engine(store=store).__aenter__()
    LLMHandler(llm).attach(engine)
    registry = ToolRegistry([lookup])
    ToolHandler(registry).attach(engine)
    return engine, ToolLoop(registry), store


async def main() -> None:
    db = Path(tempfile.mkdtemp()) / "agent.db"

    # --- process 1: starts a run, then "crashes" with the LLM request in flight
    engine, agent, store = await boot(db, HangsOnSecondCall())
    await engine.start(agent, {"task": "look up x"}, run_id="job-42")
    while not any(e.type == "aaa.tool.completed" for e in await engine.store.read("job-42")):
        await asyncio.sleep(0.02)
    await asyncio.sleep(0.2)
    await engine.abort()  # models a crash: nothing is completed, drained, or cleaned up
    await store.aclose()
    print("process 1 died with the run open")

    # --- process 2: same database, new engine; recover() re-adopts every open run
    engine, agent, store = await boot(
        db, FakeLLMClient(lambda r: text(f"result: {r.messages[-1].content}"))
    )
    handles = await engine.recover([agent])
    outcome = await handles["job-42"].result()
    print(f"process 2 recovered {len(handles)} run(s):", outcome.value["text"])
    await engine.__aexit__(None, None, None)
    await store.aclose()


asyncio.run(main())
