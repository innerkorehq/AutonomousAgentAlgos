"""The HTTP server in-process: bearer auth, a run with a filtered event stream, a human approval over HTTP, and a second tenant that cannot see any of it. Offline."""

import asyncio
import json

import httpx
from aaa_server import ApiKeyAuthenticator, InMemoryRateLimiter, ServerConfig, create_app

from aaa.algos import ToolLoop
from aaa.llm import FakeLLMClient, LLMHandler
from aaa.llm.fake import text, tool_call
from aaa.runtime import Budget, Engine
from aaa.tools import ToolHandler, ToolRegistry, tool


@tool(requires_approval=True)
def wire_money(amount: int) -> str:
    """Send money."""
    return f"sent ${amount}"


def model(request):
    last = request.messages[-1]
    if last.role == "user":
        return tool_call("wire_money", amount=250)
    return text(f"Done: {last.content}")


async def main() -> None:
    registry = ToolRegistry([wire_money])
    async with Engine() as engine:
        LLMHandler(FakeLLMClient(model)).attach(engine)
        ToolHandler(registry).attach(engine)
        app = create_app(
            ServerConfig(
                engine=engine,
                authenticator=ApiKeyAuthenticator.from_keys(
                    {  # in production: ApiKey(key_hash=...) loaded from a secret store, or JWTAuthenticator
                        "acme-key": ("acme", "ann", ["run", "approve"]),
                        "globex-key": ("globex", "gus", ["run"]),
                    }
                ),
                algorithms={"agent": ToolLoop(registry, system="Be brief.")},
                budget=Budget(
                    max_steps=20, deadline_s=60
                ),  # a request may lower this, never raise it
                rate_limiter=InMemoryRateLimiter(per_second=5, burst=20),
            )
        )
        http = httpx.ASGITransport(app=app)

        def client(key: str | None) -> httpx.AsyncClient:
            headers = {"Authorization": f"Bearer {key}"} if key else {}
            return httpx.AsyncClient(transport=http, base_url="http://aaa", headers=headers)

        async with client(None) as anon:
            print("no credentials :", (await anon.post("/v1/runs", json={})).status_code)

        async with client("acme-key") as acme:
            r = await acme.post("/v1/runs", json={"algo": "agent", "input": {"task": "pay ACME"}})
            run_id = r.json()["run_id"]
            print("started        :", r.status_code, run_id[:8] + "...")
            while True:  # the run parks until a human decides
                request_event = next(
                    (
                        e
                        for e in await engine.store.read(f"acme/{run_id}")
                        if e.type == "aaa.approval.requested"
                    ),
                    None,
                )
                if request_event:
                    break
                await asyncio.sleep(0.01)
            call_id = request_event.data["call_id"]
            print("waiting for    :", request_event.data["name"], request_event.data["arguments"])

            async with client("globex-key") as globex:
                peek = await globex.get(f"/v1/runs/{run_id}")
                print("other tenant   :", peek.status_code, peek.json()["error"]["code"])

            ok = await acme.post(
                f"/v1/runs/{run_id}/approvals", json={"call_id": call_id, "decision": "grant"}
            )
            print("approved       :", ok.status_code)
            stream = await acme.get(f"/v1/runs/{run_id}/events")
            kinds = [
                line.removeprefix("event: ")
                for line in stream.text.splitlines()
                if line.startswith("event: ")
            ]
            print("event stream   :", kinds)
            final = (await acme.get(f"/v1/runs/{run_id}")).json()
            print("outcome        :", json.dumps(final["outcome"]["value"]["text"]))


asyncio.run(main())
