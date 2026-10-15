"""A stdio MCP server made by `aaa mcp serve tests/helpers/aaa_mcp_server.py`-style factory: `build()`."""

from aaa.mcp import MCPToolServer
from aaa.tools import ToolRegistry, tool
from aaa.tools.tool import ToolError


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@tool
def boom() -> str:
    """Fails with an internal detail that must not leak."""
    raise RuntimeError("secret connection string postgres://admin:hunter2@db")


@tool
def refuse() -> str:
    """Fails politely."""
    raise ToolError("not today")


@tool(requires_approval=True)
def wire_money(amount: int) -> str:
    """Send money."""
    return f"sent {amount}"


def registry() -> ToolRegistry:
    return ToolRegistry([add, boom, refuse, wire_money])


def build() -> MCPToolServer:
    return MCPToolServer(registry())
