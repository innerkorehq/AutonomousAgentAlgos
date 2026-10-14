"""Run a ToolLoop against SQLite with an LLM that never answers, then wait to be SIGKILLed."""

import asyncio
import sys

from aaa.algos import ToolLoop
from aaa.events import SQLiteEventStore
from aaa.llm import LLMHandler, LLMRequest, LLMResponse
from aaa.runtime import Engine
from aaa.tools import ToolHandler, ToolRegistry, tool


@tool
def add(a: int, b: int) -> int:
    """add"""
    return a + b


class Hang:
    n = 0

    async def complete(self, req: LLMRequest) -> LLMResponse:
        from aaa.llm.fake import tool_call

        Hang.n += 1
        if Hang.n == 1:  # first call answers: use the tool
            return tool_call("add", a=20, b=22)
        print("READY", flush=True)  # second call hangs: killed here, mid-run
        await asyncio.sleep(3600)
        raise AssertionError

    async def embed(self, *a, **k): ...
    async def aclose(self): ...


async def main(db: str) -> None:
    reg = ToolRegistry([add])
    eng = await Engine(store=SQLiteEventStore(db)).__aenter__()
    LLMHandler(Hang()).attach(eng)
    ToolHandler(reg).attach(eng)
    h = await eng.start(ToolLoop(reg), {"task": "20+22?"}, run_id="the-run")
    print("RUN", h.run_id, flush=True)
    await asyncio.sleep(3600)


asyncio.run(main(sys.argv[1]))
