"""Choosing and naming the tools a generated server exposes (pure functions, no I/O)."""

from __future__ import annotations

import copy
import fnmatch
import re
from dataclasses import dataclass, field

from aaa.mcp.config import SpecError
from aaa.mcp.model import MCPAnnotations
from aaa.mcp.spec import TOOL_NAME, ToolsSpec
from aaa.tools.tool import Tool


@dataclass
class Entry:
    tool: Tool
    origin: str  # where it came from, for error messages ("objects myapp.tools:registry")
    tags: frozenset[str] = field(default_factory=frozenset)
    annotations: MCPAnnotations | None = (
        None  # hints the source itself knows (a module's read-only functions)
    )


def sanitize(name: str) -> str:
    """A tool name every client accepts: letters, digits, `_` and `-`, at most 64 characters."""
    return re.sub(r"[^A-Za-z0-9_-]", "_", name)[:64] or "tool"


def _matches(name: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatchcase(name, p) for p in patterns)


def with_name(tool: Tool, name: str) -> Tool:
    """The same tool under another name (a shallow copy: the function, schema and policies are shared)."""
    if tool.name == name:
        return tool
    twin = copy.copy(tool)
    twin.name = name
    return twin


def select(entries: list[Entry], policy: ToolsSpec) -> list[Entry]:
    """`include` (default everything) minus `exclude`, then those carrying at least one of `tags` when tags are given."""
    out = []
    for e in entries:
        n = e.tool.name
        if not _matches(n, policy.include) or _matches(n, policy.exclude):
            continue
        if policy.tags and not (e.tags & set(policy.tags)):
            continue
        out.append(e)
    return out


def rename(entries: list[Entry], policy: ToolsSpec) -> list[Entry]:
    """Apply `rename`, then the global `prefix`; the result must be valid, unique names."""
    unknown = set(policy.rename) - {e.tool.name for e in entries}
    if unknown:
        raise SpecError(
            f"tools.rename: no selected tool is named {sorted(unknown)} "
            f"(names are matched before renaming and after include/exclude)"
        )
    out: list[Entry] = []
    seen: dict[str, str] = {}
    for e in entries:
        name = policy.prefix + policy.rename.get(e.tool.name, e.tool.name)
        if not re.match(TOOL_NAME, name):
            name = sanitize(name)
        if name in seen:
            raise SpecError(
                f"two tools are both called {name!r} ({seen[name]} and {e.origin}); "
                "give one a different name with tools.rename or a source prefix"
            )
        seen[name] = e.origin
        out.append(Entry(with_name(e.tool, name), e.origin, e.tags, e.annotations))
    return out
