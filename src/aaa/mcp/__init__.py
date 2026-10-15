"""MCP (Model Context Protocol) client and server: use any MCP server's tools, and MCP memory servers, in AAA."""

from aaa.mcp.client import MCPClient, MCPToolInfo, mcp_tool_name, render_content
from aaa.mcp.memory import mcp_memory
from aaa.mcp.server import MCPToolServer, serve_stdio

__all__ = [
    "MCPClient",
    "MCPToolInfo",
    "MCPToolServer",
    "mcp_memory",
    "mcp_tool_name",
    "render_content",
    "serve_stdio",
]
