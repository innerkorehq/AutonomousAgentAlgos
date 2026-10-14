"""A submit-only client (an API server, a CLI): starts runs and waits for results, hosts no work.

Not executed by the test suite (needs a database). Pair it with examples/cluster_worker.py."""

import asyncio
import os

from aaa.algos import ToolLoop
from aaa.events import ClientBus
from aaa.events.postgres import PostgresEventStore
from aaa.runtime import Budget, Engine
from aaa.tools import ToolRegistry


async def main() -> None:
    store = await PostgresEventStore.create(os.environ["DATABASE_URL"], partitions=16)
    async with Engine(store=store, bus=ClientBus()) as client:
        run = await client.start(
            ToolLoop(ToolRegistry([])),  # must match the workers' algorithm name
            {"task": "Email the weekly report to the team"},
            run_id="weekly-report-2026-w42",  # idempotent: starting the same id twice is safe
            budget=Budget(max_cost=0.50, deadline_s=300),
        )
        outcome = await run.result()  # works from any process, whichever worker did the work
        print(outcome.kind.value, outcome.value)
    await store.aclose()


if __name__ == "__main__":
    asyncio.run(main())
