"""A cluster worker. Run N copies of this process against the same Postgres (or Redis) database.

Not executed by the test suite (it needs a database and runs forever); the cluster behavior is
covered by tests/unit/test_multiprocess.py, which launches real worker processes and SIGKILLs them.
Requires: pip install "autonomous-agent-architecture[postgres]"   and a reachable PostgreSQL."""

import asyncio
import os

from aaa.algos import ToolLoop
from aaa.events import DistributedBus
from aaa.events.postgres import (
    PostgresEventStore,
)  # or: from aaa.events.redis import RedisEventStore
from aaa.llm import LLMHandler, build_client, gateways
from aaa.runtime import Engine
from aaa.tools import GuardConfig, ToolHandler, ToolRegistry, tool


@tool(delivery="at_most_once")
async def send_email(to: str, subject: str) -> str:
    """Send an email."""
    return "sent"


async def main() -> None:
    store = await PostgresEventStore.create(os.environ["DATABASE_URL"], partitions=16)
    bus = DistributedBus(store.bus_backend())  # partitions, leases, rebalancing, at-least-once
    registry = ToolRegistry([send_email])
    async with Engine(store=store, bus=bus, sweep_interval_s=5) as engine:
        LLMHandler(build_client(gateways.portkey("gpt-4o-mini"))).attach(engine)
        ToolHandler(registry, guard=GuardConfig(lease_ttl_s=30)).attach(engine)
        engine.register(ToolLoop(registry))  # every worker registers the same algorithms and tools
        await asyncio.Event().wait()  # work until killed


if __name__ == "__main__":
    asyncio.run(main())
