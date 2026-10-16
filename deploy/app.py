"""The server configuration the Dockerfile runs: a starting point, not a product. Copy it, replace the model and the tools.

Environment:
  AAA_DATABASE_URL   postgresql://... (default: an in-memory store, which loses everything on restart)
  AAA_API_KEYS       JSON {"<api key>": ["<tenant>", "<subject>"]}; required (the server refuses to start without keys)
  AAA_DEMO_MODEL     "fake" (default) answers with a canned reply so the deployment can be smoke-tested without a provider
"""

from __future__ import annotations

import json
import os

from aaa_server import ApiKeyAuthenticator, InMemoryRateLimiter, ServerConfig

from aaa.algos import ToolLoop
from aaa.events import DistributedBus
from aaa.llm import FakeLLMClient, LLMHandler
from aaa.llm.fake import text
from aaa.runtime import Budget, Engine
from aaa.tools import ToolHandler, ToolRegistry


async def build() -> ServerConfig:
    keys = json.loads(os.environ.get("AAA_API_KEYS", "{}"))
    if not keys:
        raise SystemExit(
            "AAA_API_KEYS is empty: refusing to start a server nobody can authenticate to"
        )
    url = os.environ.get("AAA_DATABASE_URL")
    if url:
        from aaa.events.postgres import PostgresEventStore

        store = await PostgresEventStore.create(url)
        engine = Engine(store=store, bus=DistributedBus(store.bus_backend()))
    else:
        engine = Engine()
    registry = ToolRegistry([])
    if os.environ.get("AAA_DEMO_MODEL", "fake") != "fake":
        raise SystemExit("wire your own LLM client here (see docs/guides/llm-gateways.md)")
    LLMHandler(FakeLLMClient(lambda request: text("Hello from the container."))).attach(engine)
    ToolHandler(registry).attach(engine)
    return ServerConfig(
        engine=engine,
        authenticator=ApiKeyAuthenticator.from_keys({k: tuple(v) for k, v in keys.items()}),  # type: ignore[arg-type]
        algorithms={"agent": ToolLoop(registry, system="Be brief.")},
        budget=Budget(max_steps=20, max_tokens=50_000, deadline_s=60),
        rate_limiter=InMemoryRateLimiter(per_second=10, burst=50),
    )
