"""MCP (Model Context Protocol) client: use any MCP server's tools, and MCP memory servers, in AAA."""

from aaa.mcp.client import MCPClient, MCPToolInfo, mcp_tool_name, render_content
from aaa.mcp.memory import mcp_memory

__all__ = ["MCPClient", "MCPToolInfo", "mcp_memory", "mcp_tool_name", "render_content"]
