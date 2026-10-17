"""Build an MCP server in typed Python and expose an AAA algorithm as a tool with its own schema, budget and hints. Offline."""

import asyncio

from aaa.algos import ToolLoop
from aaa.llm import FakeLLMClient, LLMHandler
from aaa.llm.fake import text
from aaa.mcp import MCPServerBuilder, wire
from aaa.runtime import Engine
from aaa.tools import ToolRegistry, tool


@tool
def lookup_order(order_id: str) -> dict:
    """Look up an order."""
    return {"order_id": order_id, "status": "shipped"}


async def main() -> None:
    engine = Engine()
    LLMHandler(FakeLLMClient(lambda request: text("Order A1 shipped yesterday."))).attach(engine)
    support_agent = ToolLoop(ToolRegistry([lookup_order]), system="You are a support agent.")

    builder = (
        MCPServerBuilder("support-desk")
        .instructions("Use ask_support for customer questions.")
        .add_tools(ToolRegistry([lookup_order]))
        .override("lookup_order", title="Look up an order", read_only=True)
        .agent(
            "ask_support",
            algorithm=support_agent,
            description="Ask the support agent a customer question.",
            deadline_s=30,
            max_steps=10,
            max_cost=0.25,
            input_schema={
                "type": "object",
                "properties": {"question": {"type": "string"}, "customer": {"type": "string"}},
                "required": ["question"],
                "additionalProperties": False,
            },
            input_template={"task": "Customer {customer}: {question}"},
            open_world=True,
        )
        .engine(engine)
    )
    async with engine:
        server = await builder.acompile()
        for tool_def in server.list_tools():
            print(
                "tool:",
                tool_def.name,
                "|",
                tool_def.title,
                "|",
                wire.annotations_to_wire(tool_def.annotations),
            )
        answer = await server.call(
            "ask_support",
            {"question": "Where is my order?", "customer": "ann"},
            tenant="acme",
            user="ann",
        )
        print("answer:", answer.text)


asyncio.run(main())
