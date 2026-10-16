"""Throughput of a cluster on a real Postgres or Redis: runs per second and run latency vs number of workers.

    uv run python scripts/bench_cluster.py postgres postgresql://user:pw@host/db 1 2 4 --runs 200
    uv run python scripts/bench_cluster.py redis redis://localhost:6379/0 1 2 4

Each "worker" is a full Engine with its own store connections and bus, in this one process (so the machine's cores are shared:
this measures how the backend and the library's coordination behave, not what N machines would do). A fake model and a trivial
tool take no time, so the numbers are the library's and the backend's overhead per run (about five events, two tool/model steps).
The client submits all runs at once through a ClientBus and waits for every outcome."""

from __future__ import annotations

import argparse
import asyncio
import platform
import statistics
import time
import uuid

from aaa.algos import ToolLoop
from aaa.events import ClientBus, DistributedBus
from aaa.llm import FakeLLMClient, LLMHandler
from aaa.llm.fake import text, tool_call
from aaa.runtime import Engine
from aaa.tools import ToolHandler, ToolRegistry, tool


@tool
def lookup(i: int) -> str:
    """A trivial tool."""
    return f"record {i}"


def script(req):
    return tool_call("lookup", i=1) if req.messages[-1].role == "user" else text("done")


async def make_store(kind: str, target: str, prefix: str, partitions: int):
    if kind == "postgres":
        from aaa.events.postgres import PostgresEventStore

        return await PostgresEventStore.create(target, partitions=partitions)
    from aaa.events.redis import RedisEventStore

    return await RedisEventStore.create(target, partitions=partitions, prefix=prefix)


async def trial(
    kind: str, target: str, workers: int, runs: int, partitions: int, inflight: int
) -> dict[str, float]:
    prefix = (
        "bench" + uuid.uuid4().hex[:6]
    )  # a fresh keyspace per trial (Redis); Postgres rows are told apart by run id
    reg = ToolRegistry([lookup])
    engines: list[Engine] = []
    for i in range(workers):
        store = await make_store(kind, target, prefix, partitions)
        eng = Engine(
            store=store,
            bus=DistributedBus(
                store.bus_backend(), member=f"w{i}", lease_ttl_s=5, tick_s=0.2, poll_s=0.05
            ),
        )
        await eng.__aenter__()
        LLMHandler(FakeLLMClient(script)).attach(eng)
        ToolHandler(reg).attach(eng)
        eng.register(ToolLoop(reg))
        engines.append(eng)
    cstore = await make_store(kind, target, prefix, partitions)
    client = Engine(store=cstore, bus=ClientBus())
    await client.__aenter__()
    algo = ToolLoop(reg)
    await asyncio.sleep(1.0)  # let consumers claim their partitions
    tag = uuid.uuid4().hex[:6]
    t0 = time.perf_counter()
    starts: dict[str, float] = {}

    gate = asyncio.Semaphore(inflight)  # every waiting client holds a store connection: bound them

    async def one(n: int) -> float:
        async with gate:
            rid = f"{tag}-{n}"
            starts[rid] = time.perf_counter()
            h = await client.start(algo, {"task": "go"}, run_id=rid)
            out = await asyncio.wait_for(h.result(), 120)
            assert out.ok, out
            return time.perf_counter() - starts[rid]

    lat = await asyncio.gather(*(one(n) for n in range(runs)))
    wall = time.perf_counter() - t0
    for e in [client, *engines]:
        await e.__aexit__(None, None, None)
    lat.sort()
    return {
        "workers": workers,
        "runs": runs,
        "runs_per_s": runs / wall,
        "p50_ms": statistics.median(lat) * 1000,
        "p95_ms": lat[int(0.95 * (len(lat) - 1))] * 1000,
    }


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["postgres", "redis"])
    ap.add_argument("target")
    ap.add_argument("workers", nargs="*", type=int, default=[1, 2, 4])
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--partitions", type=int, default=16)
    ap.add_argument(
        "--inflight", type=int, default=20, help="runs the client keeps in flight at once"
    )
    a = ap.parse_args()
    print(
        f"# {a.kind}, {platform.machine()} {platform.system()}, python {platform.python_version()}, partitions={a.partitions}, inflight={a.inflight}"
    )
    print("| workers | runs | runs/s | p50 ms | p95 ms |\n|---|---|---|---|---|")
    for w in a.workers:
        r = await trial(a.kind, a.target, w, a.runs, a.partitions, a.inflight)
        print(
            f"| {w} | {r['runs']} | {r['runs_per_s']:.1f} | {r['p50_ms']:.0f} | {r['p95_ms']:.0f} |",
            flush=True,
        )


asyncio.run(main())
