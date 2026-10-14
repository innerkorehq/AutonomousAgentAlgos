"""Use an MCP server's tools in an agent. Needs `pip install "autonomous-agent-algos[mcp]"`.

The example writes a tiny MCP server to a temporary file and launches it as a subprocess, which is
exactly what you would do with a real one (`MCPClient.stdio("uvx", ["some-mcp-server"])`) or a remote
one (`MCPClient.http("https://.../mcp", headers={"Authorization": "Bearer ..."})`)."""

import asyncio
import sys
import tempfile
import textwrap
from pathlib import Path

from aaa.algos import ToolLoop
from aaa.llm import FakeLLMClient, LLMHandler
from aaa.llm.fake import text, tool_call
from aaa.mcp import MCPClient
from aaa.runtime import Engine
from aaa.tools import ToolHandler, ToolRegistry

SERVER = textwrap.dedent(
    """
    try:  # mcp 2.x
        from mcp.server.mcpserver import MCPServer as Server
        from mcp.types import ToolAnnotations
        READ_ONLY = ToolAnnotations(read_only_hint=True)
    except ImportError:  # mcp 1.x
        from mcp.server.fastmcp import FastMCP as Server
        from mcp.types import ToolAnnotations
        READ_ONLY = ToolAnnotations(readOnlyHint=True)

    server = Server("weather")

    @server.tool(annotations=READ_ONLY)
    def forecast(city: str, days: int = 2) -> str:
        "Forecast for a city."
        return f"{city}: sunny for the next {days} days"

    @server.tool()
    def reset_station(station: str) -> str:
        "Reset a weather station. No annotations: the MCP spec says to assume it is destructive."
        return f"{station} reset"

    server.run()
    """
)


def scripted_model(request):  # stands in for a real LLM
    last = request.messages[-1]
    if last.role == "user":
        return tool_call("weather__forecast", city="Lisbon", days=3)
    return text(f"The forecast says: {last.content}")


async def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "weather_server.py"
        script.write_text(SERVER)
        async with MCPClient.stdio(sys.executable, [str(script)], name="weather") as weather:
            tools = await weather.load_tools(approval="destructive")  # a human approves risky tools
            for t in tools:
                print(f"{t.name:24} delivery={t.delivery:13} approval={t.requires_approval}")
            registry = ToolRegistry(tools)
            async with Engine() as engine:
                LLMHandler(FakeLLMClient(scripted_model)).attach(engine)
                ToolHandler(registry).attach(engine)
                outcome = await engine.run(ToolLoop(registry), {"task": "weather in Lisbon?"})
            print(outcome.value["text"])


asyncio.run(main())
