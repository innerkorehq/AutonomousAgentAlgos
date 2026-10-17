"""Generate an MCP server from a configuration instead of writing one: choose, rename and describe the tools you expose. Offline."""

import asyncio
import json
import tempfile
from pathlib import Path

from pydantic import BaseModel

from aaa.mcp.compile import acompile_spec, describe
from aaa.mcp.config import load_spec
from aaa.tools import ToolRegistry, tool


class Weather(BaseModel):
    city: str
    celsius: float


@tool(delivery="at_least_once")
def forecast(city: str) -> Weather:
    """Today's forecast for a city."""
    return Weather(city=city, celsius=21.5)


@tool
def shout(text: str) -> str:
    """Upper-case some text."""
    return text.upper()


@tool(requires_approval=True)
def erase_history() -> str:
    """Delete all stored history."""
    return "erased"


tools = ToolRegistry([forecast, shout, erase_history])

CONFIG = """
[server]
name = "weather-desk"
instructions = "Answer weather questions with the forecast tool."

[[sources]]
type = "objects"
ref = "__main__:tools"          # any AAA objects: a ToolRegistry, an MCPToolServer, a list of tools, or a factory

[tools]
exclude = ["shout"]             # never expose these
prefix = "desk_"                # exposed names get this prefix

[tools.overrides.desk_forecast]
title = "Weather forecast"
annotations = { read_only = true, open_world = true }
"""


async def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "server.toml"
        path.write_text(CONFIG)
        path.chmod(0o644)
        core = await acompile_spec(load_spec(path))
    shown = describe(core)
    for t in shown["tools"]:
        print("offers:", t["name"], "|", t["title"], "|", t["annotations"])
    print("hidden (need approval):", shown["hidden"])
    result = await core.call("desk_forecast", {"city": "Oslo"})
    print("structured result:", json.dumps(result.structured))


asyncio.run(main())
