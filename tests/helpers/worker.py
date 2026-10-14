"""A cluster worker process: DistributedBus + Engine + LLM/tool handlers, forever."""

import asyncio
import os
import sys

from aaa.algos import ToolLoop
from aaa.events import DistributedBus
from aaa.llm import FakeLLMClient, LLMHandler
from aaa.llm.fake import text, tool_call
from aaa.runtime import Engine
from aaa.tools import GuardConfig, ToolHandler, ToolRegistry, tool

kind, target, member, prefix, effects = sys.argv[1:6]


SECONDS = float(os.environ.get("AAA_TOOL_SECONDS", "0.3"))
DELIVERY = os.environ.get("AAA_TOOL_DELIVERY", "at_least_once")


@tool(delivery=DELIVERY)
async def work(n: int) -> int:
    """do (slow) work; starts go to <effects>.starts, completions to <effects>"""
    fd = os.open(effects + ".starts", os.O_WRONLY | os.O_APPEND | os.O_CREAT)
    os.write(fd, f"{n} {member}\n".encode())
    os.close(fd)
    await asyncio.sleep(SECONDS)
    fd = os.open(effects, os.O_WRONLY | os.O_APPEND | os.O_CREAT)
    os.write(fd, f"{n} {member}\n".encode())
    os.close(fd)
    return n * 2


def script(req):
    last = req.messages[-1]
    if last.role == "user":
        return tool_call("work", n=int("".join(c for c in last.content if c.isdigit())))
    return text(f"result {last.content}")


async def main() -> None:
    if kind == "postgres":
        from aaa.events.postgres import PostgresEventStore

        store = await PostgresEventStore.create(target, partitions=8)
    else:
        from aaa.events.redis import RedisEventStore

        store = await RedisEventStore.create(target, partitions=8, prefix=prefix)
    bus = DistributedBus(
        store.bus_backend(), member=member, lease_ttl_s=1.5, tick_s=0.2, poll_s=0.2
    )
    reg = ToolRegistry([work])
    eng = Engine(store=store, bus=bus, sweep_interval_s=1.0)
    await eng.__aenter__()
    LLMHandler(FakeLLMClient(script)).attach(eng)
    ToolHandler(
        reg,
        guard=GuardConfig(
            lease_ttl_s=float(os.environ.get("AAA_LEASE_TTL", "1.5")),
            heartbeat_s=0.3,
            wait_poll_s=0.2,
        ),
    ).attach(eng)
    eng.register(ToolLoop(reg))
    print("READY", flush=True)
    await asyncio.sleep(3600)


asyncio.run(main())
