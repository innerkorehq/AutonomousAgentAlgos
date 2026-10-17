"""One configuration for every kind of tool: a Python module, an OpenAPI document, another MCP server, a static resource and a prompt,
split over an included file, with a per-tool policy and a production overlay. Offline."""

import asyncio
import json
import os
import sys
import tempfile
import textwrap
from pathlib import Path

from aaa.mcp.compile import acompile_spec
from aaa.mcp.config import load_spec

MODULE = '''
__all__ = ["word_count"]

def word_count(text: str) -> int:
    """Count the words in a text."""
    return len(text.split())
'''
UPSTREAM = """
import asyncio
from aaa.mcp import MCPToolServer, serve_stdio
from aaa.tools import ToolRegistry, tool

@tool
def shout(text: str) -> str:
    "Upper-case a text."
    return text.upper()

asyncio.run(serve_stdio(MCPToolServer(ToolRegistry([shout]), name="upstream")))
"""
OPENAPI = {
    "openapi": "3.0.3",
    "info": {"title": "Orders", "version": "1"},
    "servers": [{"url": "https://api.example.com/v1"}],
    "paths": {
        "/orders/{id}": {
            "get": {
                "operationId": "getOrder",
                "parameters": [
                    {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
                ],
                "responses": {"200": {"description": "the order"}},
            }
        }
    },
}
ORDERS = """
[[sources]]
type = "openapi"
document = "orders.json"
allowed_hosts = ["api.example.com"]
"""
SERVER = f"""
include = ["orders.toml"]

[server]
name = "everything"

[[sources]]
type = "module"
module = "textkit"
path = "."
allow = ["word_count"]
no_approval = ["word_count"]

[[sources]]
type = "mcp"
name = "up"
transport = "stdio"
command = "{Path(sys.executable).as_posix()}"
args = ["@DIR@/upstream.py"]
env_from = ["PYTHONPATH"]  # only what is named reaches the child process
approval = "none"
prefix = "up_"

[[resources]]
type = "static"
name = "readme"
text = "One server for everything."

[[prompts]]
name = "summarize"
description = "Summarize a text."
arguments = [{{ name = "text", required = true }}]
messages = [{{ role = "user", text = "Summarize: {{text}}" }}]

[[tools.policies]]
match = ["word_count", "up_*"]
timeout_s = 10
rate_limit = {{ per_minute = 120 }}
"""
PROD = """
[server]
title = "Everything (production)"
"""


async def main() -> None:
    os.environ.setdefault("PYTHONPATH", "")
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        (d / "textkit.py").write_text(textwrap.dedent(MODULE))
        (d / "upstream.py").write_text(UPSTREAM)
        (d / "orders.json").write_text(json.dumps(OPENAPI))
        (d / "orders.toml").write_text(ORDERS)
        (d / "server.toml").write_text(SERVER.replace("@DIR@", d.as_posix()))
        (d / "prod.toml").write_text(PROD)
        spec = load_spec(d / "server.toml", environ={}, overlays=[d / "prod.toml"])
        async with await acompile_spec(spec) as core:
            print("title:", core.title)
            print("tools:", sorted(t.name for t in core.list_tools()))
            print("resources:", [r.name for r in core.list_resources()])
            print("prompts:", [p.name for p in core.list_prompts()])
            print(
                "word_count:", (await core.call("word_count", {"text": "one server for all"})).text
            )
            print("up_shout:", (await core.call("up_shout", {"text": "hello"})).text)
            print("policies:", sorted(core.tool_policy))


asyncio.run(main())
