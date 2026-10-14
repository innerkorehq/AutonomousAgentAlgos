"""Tenancy, redaction and erasure on one engine. Runs offline with a scripted model."""

import asyncio
from datetime import timedelta

from aaa.algos import ToolLoop
from aaa.llm import FakeLLMClient, LLMHandler
from aaa.llm.fake import text, tool_call
from aaa.privacy import standard_redactor
from aaa.runtime import Engine, RetentionPolicy
from aaa.tools import ToolHandler, ToolRegistry, tool


@tool
def lookup_customer(customer_id: str) -> str:
    """Fetch a customer record."""
    return f"{customer_id}: ann@example.com, card 4111 1111 1111 1111"


def scripted_model(request):
    last = request.messages[-1]
    if last.role == "user":
        return tool_call("lookup_customer", customer_id="C-7")
    return text(f"found {last.content}")


async def main() -> None:
    registry = ToolRegistry([lookup_customer])
    engine = Engine(
        redactor=standard_redactor(),  # scrub e-mails, cards, keys... before anything is stored
        require_tenant=True,  # no run may exist without an owner
        retention=[
            RetentionPolicy(older_than=timedelta(days=365), tenant="acme"),  # acme: keep a year
            RetentionPolicy(older_than=timedelta(days=30)),  # everyone else: 30 days, then scrub
        ],
    )
    async with engine:
        LLMHandler(FakeLLMClient(scripted_model)).attach(engine)
        ToolHandler(registry).attach(engine)
        algo = ToolLoop(registry)

        acme, globex = engine.tenant("acme"), engine.tenant("globex")
        outcome = await acme.run(algo, {"task": "look up C-7"}, run_id="ticket-1")
        await globex.run(algo, {"task": "look up C-7"}, run_id="ticket-1")  # same id, no clash
        print("answer:", outcome.value["text"])

        log = await engine.store.read("acme/ticket-1")
        print("tenants on the log:", sorted({e.tenant or "-" for e in log}))
        print("stored e-mail visible:", "ann@example.com" in str([e.data for e in log]))

        try:
            await acme.handle("globex/ticket-1")
        except KeyError as e:
            print("acme asking for globex's run:", e)

        report = await acme.erase("ticket-1", reason="customer request #42")
        log = await engine.store.read("acme/ticket-1")
        print(f"erased {report.events} events; last event: {log[-1].type}")
        print("payload left:", "found" in str([e.data for e in log]))
        print("still finished:", (await (await acme.handle("ticket-1")).result()).ok)
        print(
            "globex untouched:",
            "found" in str([e.data for e in await engine.store.read("globex/ticket-1")]),
        )


asyncio.run(main())
