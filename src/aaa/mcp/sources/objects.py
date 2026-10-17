"""Tools from AAA objects: a `ToolRegistry`, an `MCPToolServer`, a `Tool`, a list of tools, or a function returning one."""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Mapping
from typing import Any

from aaa.mcp.config import SpecError
from aaa.tools.tool import Tool, ToolRegistry


def normalize(obj: Any, where: str) -> list[Tool]:
    """The tools in `obj`, whatever shape it has; anything else is a configuration error that says what was found."""
    from aaa.mcp.server import MCPToolServer

    if isinstance(obj, Tool):
        return [obj]
    if isinstance(obj, ToolRegistry):
        return [t for n in obj.names() if (t := obj.get(n)) is not None]
    if isinstance(obj, MCPToolServer):
        return normalize(obj.registry, where)
    if isinstance(obj, Mapping):
        return normalize(list(obj.values()), where)
    if isinstance(obj, list | tuple) and all(isinstance(t, Tool) for t in obj):
        return list(obj)
    raise SpecError(
        f"{where}: expected a ToolRegistry, an MCPToolServer, a Tool or a list of tools, got {type(obj).__name__}"
    )


async def load_ref(ref: str) -> list[Tool]:
    """Import `package.module:attribute` and return its tools. Importing runs the module's code: callers check the kill-switch first."""
    module, _, attr = ref.partition(":")
    try:
        mod = importlib.import_module(module)
    except ImportError as e:
        raise SpecError(
            f"sources: cannot import {module!r} for {ref!r} ({type(e).__name__}: {e})"
        ) from None
    try:
        obj = mod
        for part in attr.split("."):
            obj = getattr(obj, part)
    except AttributeError:
        raise SpecError(f"sources: module {module!r} has no attribute {attr!r}") from None
    try:
        if inspect.isroutine(obj) and not isinstance(
            obj, Tool
        ):  # a plain function or method; classes are never instantiated
            obj = obj()
        if inspect.isawaitable(obj):
            obj = await obj
    except Exception as e:  # noqa: BLE001 - the user's factory failed; say which one, not a stack from our internals
        raise SpecError(f"sources: {ref} raised {type(e).__name__}: {str(e)[:200]}") from None
    return normalize(obj, ref)
