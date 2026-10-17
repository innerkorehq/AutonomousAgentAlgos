"""Turn the functions of a Python module into MCP tools: an explicit list, schemas from type hints, approval by default. Offline."""

import asyncio
import sys
import tempfile
import textwrap
from pathlib import Path

from aaa.mcp import MCPServerBuilder

MODULE = '''
__all__ = ["word_count", "reverse", "delete_file"]

def word_count(text: str) -> int:
    """Count the words in a text.

    Args:
        text: the text to count
    """
    return len(text.split())

def reverse(text: str) -> str:
    """Reverse a text."""
    return text[::-1]

def delete_file(path: str) -> str:
    """Delete a file (never exposed unless approved)."""
    raise RuntimeError("not reached")
'''


async def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "text_tools.py").write_text(textwrap.dedent(MODULE))
        server = await (
            MCPServerBuilder("text")
            .add_module("text_tools", path=tmp, no_approval=["word_count"], read_only=["reverse"])
            .acompile()
        )
        sys.modules.pop("text_tools", None)
    for t in server.list_tools():
        print("tool:", t.name, "|", t.description, "|", sorted(t.input_schema["properties"]))
    print(
        "description of 'text':",
        server.list_tools()[0].input_schema["properties"]["text"]["description"],
    )
    print("hidden (need approval):", server.hidden)
    print("word_count ->", (await server.call("word_count", {"text": "one two three"})).text)


asyncio.run(main())
