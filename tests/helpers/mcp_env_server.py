"""A real MCP server (official SDK) for upstream tests: it reports its own process, so tests can see what environment it was given and
whether it is still alive. `python tests/helpers/mcp_env_server.py`."""

from __future__ import annotations

import os

try:  # mcp >= 2
    from mcp.server.mcpserver import MCPServer as _Server
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server  # type: ignore[no-redef]


def build():
    server = _Server("envreport")

    @server.tool()
    def env_names() -> str:
        """The names of the environment variables this process has."""
        return ",".join(sorted(os.environ))

    @server.tool()
    def env_value(name: str) -> str:
        """The value of one environment variable (or the empty string)."""
        return os.environ.get(name, "")

    @server.tool()
    def pid() -> str:
        """This process id."""
        return str(os.getpid())

    @server.tool()
    def echo(text: str) -> str:
        """Echo."""
        return text

    return server


if __name__ == "__main__":
    build().run()
