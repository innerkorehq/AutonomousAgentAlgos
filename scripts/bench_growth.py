"""How does a long agent run scale? Measures log size and time vs number of turns.

    uv run python scripts/bench_growth.py [turns ...]

Uses a fake LLM (no network) and SQLite, so the numbers isolate the library's own overhead:
bytes written to the event log, bytes of llm.requested events, snapshot bytes, wall time."""

from __future__ import annotations

import asyncio
import os
import sqlite3
import sys
import tempfile
import time

from aaa.algos import ToolLoop
from aaa.events import SQLiteEventStore
from aaa.llm import FakeLLMClient, LLMHandler
from aaa.llm.fake import text, tool_call
from aaa.runtime import Engine
from aaa.tools import ToolHandler, ToolRegistry, tool

PAYLOAD = "x" * 400  # each tool result is ~400 bytes, like a small search result


@tool
def lookup(i: int) -> str:
    """Return a small record."""
    return f"{i}:{PAYLOAD}"


async def measure(turns: int) -> dict[str, float]:
    reg = ToolRegistry([lookup])
    n = {"calls": 0}

    def script(req):
        n["calls"] += 1
        return tool_call("lookup", i=n["calls"]) if n["calls"] <= turns else text("done")

    path = os.path.join(tempfile.mkdtemp(), "g.db")
    store = SQLiteEventStore(path)
    t = time.perf_counter()
    async with Engine(store=store) as eng:
        LLMHandler(FakeLLMClient(script)).attach(eng)
        ToolHandler(reg).attach(eng)
        out = await eng.run(ToolLoop(reg, max_turns=turns + 5), {"task": "go"})
    elapsed = time.perf_counter() - t
    assert out.ok, out
    await store.aclose()
    c = sqlite3.connect(path)
    total, events = c.execute(
        "SELECT COALESCE(SUM(LENGTH(body)),0), COUNT(*) FROM events"
    ).fetchone()
    req = c.execute(
        "SELECT COALESCE(SUM(LENGTH(body)),0) FROM events WHERE type='aaa.llm.requested'"
    ).fetchone()[0]
    snap = c.execute("SELECT COALESCE(SUM(LENGTH(state)),0) FROM snapshots").fetchone()[0]
    return {
        "turns": turns,
        "events": events,
        "log_KB": total / 1024,
        "requested_KB": req / 1024,
        "snapshots_KB": snap / 1024,
        "seconds": elapsed,
    }


async def main() -> None:
    sizes = [int(a) for a in sys.argv[1:]] or [25, 50, 100, 200]
    print(
        f"{'turns':>6} {'events':>7} {'log KB':>9} {'requested KB':>13} {'snapshot KB':>12} {'seconds':>8}"
    )
    for t in sizes:
        r = await measure(t)
        print(
            f"{r['turns']:>6.0f} {r['events']:>7.0f} {r['log_KB']:>9.0f} {r['requested_KB']:>13.0f} "
            f"{r['snapshots_KB']:>12.0f} {r['seconds']:>8.2f}"
        )


asyncio.run(main())
