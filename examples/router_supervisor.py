"""Pick one specialist (Router), or let an LLM delegate to several workers in parallel (Supervisor)."""

import asyncio

from aaa.algos import Route, Router, Supervisor, ToolLoop
from aaa.llm import FakeLLMClient, LLMHandler, LLMResponse, Message, ToolCall
from aaa.llm.fake import text
from aaa.runtime import Engine
from aaa.tools import ToolHandler, ToolRegistry

empty = ToolRegistry()
billing = ToolLoop(empty, name="billing", system="You are billing support.")
tech = ToolLoop(empty, name="tech", system="You are technical support.")
researcher = ToolLoop(empty, name="researcher", system="You research.")
writer = ToolLoop(empty, name="writer", system="You write.")


def model(request):
    system = request.messages[0].content if request.messages[0].role == "system" else ""
    last = request.messages[-1]
    if "billing support" in system:
        return text("Refund issued.")
    if "technical support" in system:
        return text("Try turning it off and on again.")
    if "You research" in system:
        return text("Findings: AAA is event-driven.")
    if "You write" in system:
        return text("Draft: AAA logs everything as events.")
    if "coordinate" in system:  # the supervisor
        if last.role == "user":
            return LLMResponse(
                message=Message(
                    role="assistant",
                    tool_calls=[
                        ToolCall(
                            id="1", name="delegate_researcher", arguments={"task": "research AAA"}
                        ),
                        ToolCall(
                            id="2", name="delegate_writer", arguments={"task": "write about AAA"}
                        ),
                    ],
                )
            )
        return text(
            "Report ready:\n" + "\n".join(m.content for m in request.messages if m.role == "tool")
        )
    return text("?")


async def main() -> None:
    router = Router(
        {"billing": Route(billing, "invoices, refunds"), "tech": Route(tech, "bugs, crashes")},
        classifier=lambda inp: "billing" if "invoice" in inp["task"] else None,  # cheap rules first
        default="tech",  # if the (optional) LLM classifier picks something unknown
    )
    supervisor = Supervisor(
        {"researcher": (researcher, "finds facts"), "writer": (writer, "drafts text")}
    )
    async with Engine() as engine:
        LLMHandler(FakeLLMClient(model)).attach(engine)
        ToolHandler(supervisor.tools).attach(engine)
        print((await engine.run(router, {"task": "my invoice is wrong"})).value["text"])
        print((await engine.run(supervisor, {"task": "Write a short report on AAA"})).value["text"])


asyncio.run(main())
