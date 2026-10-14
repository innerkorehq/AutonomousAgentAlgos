"""Human-in-the-loop: a risky tool waits for an out-of-band approval event."""

import asyncio

from aaa.algos import ToolLoop
from aaa.llm import FakeLLMClient, LLMHandler
from aaa.llm.fake import text, tool_call
from aaa.runtime import Engine
from aaa.tools import ToolHandler, ToolRegistry, tool


@tool(requires_approval=True)  # enforced by ToolHandler, outside the model's control
def wire_money(amount: int, to: str) -> str:
    """Send money."""
    return f"sent ${amount} to {to}"


def model(request):
    last = request.messages[-1]
    if last.role == "user":
        return tool_call("wire_money", amount=250, to="ACME")
    return text(last.content)


async def main() -> None:
    registry = ToolRegistry([wire_money])
    async with Engine() as engine:
        LLMHandler(FakeLLMClient(model)).attach(engine)
        ToolHandler(registry).attach(engine)
        run = await engine.start(ToolLoop(registry), {"task": "Pay ACME $250"})

        async def pending():  # in a real app this is a webhook / UI / Slack button
            while True:
                for e in await engine.store.read(run.run_id):
                    if e.type == "aaa.approval.requested":
                        return e
                await asyncio.sleep(0.02)

        request = await asyncio.wait_for(pending(), 5)
        print("waiting for approval of:", request.data["name"], request.data["arguments"])
        await run.signal("aaa.approval.granted", {"call_id": request.data["call_id"]})
        # or: await run.signal("aaa.approval.denied", {"call_id": ..., "reason": "too large"})
        outcome = await run.result()
    print(outcome.value["text"])


asyncio.run(main())
