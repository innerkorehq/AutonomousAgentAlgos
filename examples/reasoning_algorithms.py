"""Four ways to spend more model calls for a better answer: Reflexion, Debate, Swarm handoffs and Tree-of-thoughts. Scripted model, offline."""

import asyncio
import json

from aaa.algos import Agent, Debate, Reflexion, Swarm, TreeOfThoughts
from aaa.llm import FakeLLMClient, LLMHandler
from aaa.llm.fake import text, tool_call
from aaa.runtime import Engine
from aaa.tools import ToolHandler, ToolRegistry, tool
from aaa.validate import ValidationHandler


def system_of(request) -> str:
    return next((m.content or "" for m in request.messages if m.role == "system"), "")


def user_of(request) -> str:
    return next((m.content or "" for m in reversed(request.messages) if m.role == "user"), "")


@tool
def refund(order: str) -> str:
    """Refund an order."""
    return f"refunded {order}"


async def reflexion() -> None:
    def model(request):
        if "reviewing a failed attempt" in system_of(request):
            return text("State the unit.")
        learned = any("State the unit" in (m.content or "") for m in request.messages)
        return text("42 km" if learned else "42")

    async with Engine() as engine:
        LLMHandler(FakeLLMClient(model)).attach(engine)
        ValidationHandler({"unit": lambda answer, ctx: ("km" in answer, "no unit given")}).attach(
            engine
        )
        out = await engine.run(Reflexion(validator="unit"), {"task": "How far is it?"})
    print(
        "reflexion:",
        out.value["candidate"],
        "after",
        out.value["trials"],
        "trials; lesson:",
        out.value["reflections"],
    )


async def debate() -> None:
    first = {"A": "12", "B": "10", "C": "12"}

    def model(request):
        if "judge of a debate" in system_of(request):
            return text("Answer: 12")
        agent = next(a for a in "ABC" if f"[{a}]" in system_of(request))
        revised = "Other agents answered" in user_of(request)
        return text(f"Answer: {'12' if revised else first[agent]}")

    async with Engine() as engine:
        LLMHandler(FakeLLMClient(model)).attach(engine)
        personas = ["[A] careful", "[B] fast", "[C] skeptical"]
        out = await engine.run(Debate(agents=3, rounds=1, personas=personas), {"task": "6 x 2?"})
    print("debate:", out.value["answer"], "from rounds", out.value["answers"])


async def swarm() -> None:
    registry = ToolRegistry([refund])
    agents = {
        "triage": Agent("You route customers.", handoffs=["billing"]),
        "billing": Agent(
            "You handle billing.", tools=["refund"], description="Refunds and charges."
        ),
    }

    def model(request):
        offered = {t.name for t in request.tools}
        last = request.messages[-1]
        if "transfer_to_billing" in offered:
            return tool_call("transfer_to_billing")
        if last.role == "tool" and "Transferred" in (last.content or ""):
            return tool_call("refund", "c2", order="A1")
        return text(f"Done: {last.content}")

    async with Engine() as engine:
        LLMHandler(FakeLLMClient(model)).attach(engine)
        ToolHandler(registry).attach(engine)
        out = await engine.run(Swarm(agents, registry, start="triage"), {"task": "Refund order A1"})
    print(
        "swarm:",
        out.value["text"],
        "| handoffs:",
        out.value["handoffs"],
        "| answered by:",
        out.value["agent"],
    )


async def tree() -> None:
    def model(request):
        if "Rate how promising" in system_of(request):
            return text(json.dumps({"score": 10 if "FINAL" in user_of(request) else 6}))
        steps = user_of(request).count("\n1. ") + user_of(request).count("\n2. ")
        return text("FINAL: (8 / (3 - 8 / 3))" if steps >= 2 else "consider 8 / 3")

    algorithm = TreeOfThoughts(depth=3, width=2, branching=2)
    async with Engine() as engine:
        LLMHandler(FakeLLMClient(model)).attach(engine)
        out = await engine.run(algorithm, {"task": "Make 24 from 3, 3, 8, 8"})
    print(
        "tree:",
        out.value["answer"],
        "| score",
        out.value["score"],
        "| at most",
        algorithm.max_calls,
        "model calls",
    )


async def main() -> None:
    await reflexion()
    await debate()
    await swarm()
    await tree()


asyncio.run(main())
