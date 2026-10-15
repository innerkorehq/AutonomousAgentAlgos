"""Offer your tools to any MCP client, safely: approval-gated tools stay hidden, failures never leak internals, and an agent can be run as one budgeted tool. The protocol core runs offline; `serve_stdio(server)` or `aaa mcp serve` puts it on the wire."""

import asyncio

from aaa.algos import ToolLoop
from aaa.llm import FakeLLMClient, LLMHandler
from aaa.llm.fake import text
from aaa.mcp import MCPToolServer
from aaa.runtime import Budget, Engine
from aaa.tools import ToolRegistry, tool


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@tool
def lookup(key: str) -> str:
    """Look something up (always fails, with an internal detail)."""
    raise RuntimeError("could not reach postgres://admin:hunter2@db")


@tool(requires_approval=True)
def wire_money(amount: int) -> str:
    """Send money."""
    return f"sent {amount}"


async def main() -> None:
    registry = ToolRegistry([add, lookup, wire_money])
    async with Engine() as engine:
        LLMHandler(FakeLLMClient(lambda request: text("Paris."))).attach(engine)
        server = MCPToolServer(
            registry,
            engine=engine,
            algorithms={"qa": ToolLoop(ToolRegistry([]))},
            agent_budget=Budget(max_steps=5, deadline_s=30),
        )
        print("offered:", [t.name for t in server.list_tools()])
        print("add:", (await server.call("add", {"a": 2, "b": 3})).text)
        print("bad args:", (await server.call("add", {"a": "x", "b": 3})).is_error)
        print("lookup:", (await server.call("lookup", {"key": "k"})).text)
        hidden = await server.call("wire_money", {"amount": 1000})
        print("wire_money:", hidden.text, "| error:", hidden.is_error)
        answer = await server.call(
            "run_agent", {"algo": "qa", "input": {"task": "capital of France?"}}
        )
        print("run_agent:", answer.text)


asyncio.run(main())
