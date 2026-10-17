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


# -- tools for the generator tests ---------------------------------------------------------------
from pydantic import BaseModel


class Point(BaseModel):
    x: float
    y: float


@tool(delivery="at_least_once")
def locate(name: str) -> Point:
    """Find a place (its return type is an object, so it has an output schema)."""
    return Point(x=1.5, y=2.5)


@tool
def broken_shape() -> Point:
    """Claims to return a Point but returns something else."""
    return {"x": "not a number"}  # type: ignore[return-value]


@tool
def shout(text: str) -> str:
    """Upper-case some text."""
    return text.upper()


@tool(requires_approval=True)
def delete_all() -> str:
    """Delete everything."""
    return "gone"


def rich_registry() -> ToolRegistry:
    return ToolRegistry([add, locate, broken_shape, shout, delete_all])
