"""Tools with side effects: delivery policy, idempotency keys, and fencing, all configurable."""

import asyncio

from aaa.algos import ToolLoop
from aaa.llm import FakeLLMClient, LLMHandler
from aaa.llm.fake import text, tool_call
from aaa.runtime import Engine
from aaa.tools import GuardConfig, ToolContext, ToolHandler, ToolRegistry, tool


@tool(  # a payment must never run twice: report a lost attempt instead of retrying it
    delivery="at_most_once",
    idempotency_key=lambda args: f"refund:{args['order_id']}",  # a business key, not a call id
)
async def refund(order_id: str, ctx: ToolContext) -> str:
    """Refund an order."""
    # Pass the key to the payment provider so even a repeated attempt cannot repeat the effect:
    #   await stripe.refunds.create(..., idempotency_key=ctx.idempotency_key)
    return (
        f"refunded {order_id} (key={ctx.idempotency_key}, attempt={ctx.attempt}, fence={ctx.fence})"
    )


@tool  # inherits the handler default: re-run a lost call, at most 3 attempts
async def search(query: str) -> str:
    """Search."""
    return f"results for {query}"


def model(request):
    last = request.messages[-1]
    return tool_call("refund", order_id="A-100") if last.role == "user" else text(last.content)


async def main() -> None:
    registry = ToolRegistry([refund, search])
    guard = GuardConfig(lease_ttl_s=30, max_attempts=3)  # handler-wide default; tools override it
    handler = ToolHandler(registry, guard=guard)
    print(
        "refund  ->",
        handler.config_for(refund).delivery.value,
        "| max_attempts =",
        handler.config_for(refund).max_attempts,
    )
    print(
        "search  ->",
        handler.config_for(search).delivery.value,
        "| max_attempts =",
        handler.config_for(search).max_attempts,
    )
    async with Engine() as engine:
        LLMHandler(FakeLLMClient(model)).attach(engine)
        handler.attach(engine)
        outcome = await engine.run(ToolLoop(registry), {"task": "refund order A-100"})
    print(outcome.value["text"])


asyncio.run(main())
