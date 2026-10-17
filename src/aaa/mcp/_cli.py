"""`aaa mcp serve|check|inspect`: the command line of the MCP server generator.

    aaa mcp serve --config server.yaml                    # a generated server over stdio
    aaa mcp serve --source myapp.tools:registry --exclude 'admin_*' --tenant acme
    aaa mcp serve myapp.tools:build                       # the older form: a factory returning a ToolRegistry/MCPToolServer
    aaa mcp check --config server.yaml                    # validate and compile; report what would be exposed
    aaa mcp inspect --config server.yaml                  # print exactly what a client (and its model) will be shown

Flags override the environment (`AAA_MCP__*`), which overrides the file. stdout carries the protocol while serving, so every
message from `serve` goes to stderr."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any

from aaa.mcp.config import SpecError, load_spec


def add_arguments(mc: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--config", help="a .yaml, .yml, .json or .toml server configuration")
        p.add_argument(
            "--source",
            action="append",
            metavar="MODULE:ATTR",
            help="tools from AAA objects (repeatable); replaces the file's sources",
        )
        p.add_argument(
            "--include",
            action="append",
            metavar="GLOB",
            help="expose only matching tool names (repeatable)",
        )
        p.add_argument(
            "--exclude",
            action="append",
            metavar="GLOB",
            help="never expose matching tool names (repeatable)",
        )
        p.add_argument(
            "--tag", action="append", help="expose only tools carrying this source tag (repeatable)"
        )
        p.add_argument("--prefix", help="prefix every exposed tool name")
        p.add_argument("--name", help="server name")
        p.add_argument("--title", help="server title")
        p.add_argument("--instructions", help="instructions shown to the client")
        p.add_argument("--tenant", help="tenant the calls are for")
        p.add_argument("--user", help="user the calls are for")
        p.add_argument(
            "--expose-approval-tools",
            action="store_true",
            default=None,
            help="also offer tools that require approval (nothing is approved for you)",
        )
        p.add_argument("--max-result-chars", type=int)
        p.add_argument(
            "--no-import-sources", action="store_true", help="refuse any source that imports code"
        )

    sv = mc.add_parser("serve", help="serve tools over MCP stdio")
    sv.add_argument(
        "factory",
        nargs="?",
        help="older form: module:function returning a ToolRegistry or MCPToolServer",
    )
    common(sv)
    ck = mc.add_parser("check", help="validate a configuration and report what it would expose")
    common(ck)
    ck.add_argument(
        "--no-compile", action="store_true", help="only validate the file; do not import anything"
    )
    common(mc.add_parser("inspect", help="print what clients will be shown"))


def overlay_of(a: argparse.Namespace) -> dict[str, Any]:
    o: dict[str, Any] = {}

    def put(path: str, value: Any) -> None:
        if value is None:
            return
        node = o
        *parents, last = path.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[last] = value

    if a.source:
        put("sources", [{"type": "objects", "ref": r} for r in a.source])
    put("tools.include", a.include)
    put("tools.exclude", a.exclude)
    put("tools.tags", a.tag)
    put("tools.prefix", a.prefix)
    put("tools.expose_approval_tools", True if a.expose_approval_tools else None)
    put("server.name", a.name)
    put("server.title", a.title)
    put("server.instructions", a.instructions)
    put("tenancy.tenant", a.tenant)
    put("tenancy.user", a.user)
    put("limits.max_result_chars", a.max_result_chars)
    return o


def _fail(msg: str) -> int:
    print(f"error: {msg}", file=sys.stderr)
    return 2


async def _legacy_serve(spec: str) -> int:
    import importlib
    import inspect

    from aaa.mcp.server import MCPToolServer, serve_stdio
    from aaa.tools import ToolRegistry

    module, _, attr = spec.partition(":")
    if not module or not attr:
        return _fail("expected module:factory, e.g. myapp.tools:build")
    sys.path.insert(0, "")
    obj = getattr(importlib.import_module(module), attr)
    obj = obj() if callable(obj) else obj
    if inspect.isawaitable(obj):
        obj = await obj
    if isinstance(obj, ToolRegistry):
        obj = MCPToolServer(obj)
    if not isinstance(obj, MCPToolServer):
        return _fail(f"{spec} must return a ToolRegistry or MCPToolServer")
    await serve_stdio(obj)
    return 0


async def run(a: argparse.Namespace) -> int:
    from aaa.mcp.compile import acompile_spec, describe
    from aaa.mcp.server import serve_stdio

    sys.path.insert(0, "")
    if a.sub == "serve" and a.factory and not a.config:
        return await _legacy_serve(a.factory)
    try:
        spec = load_spec(a.config, overlay=overlay_of(a))
        if a.sub == "check" and a.no_compile:
            print(
                f"ok: configuration is valid ({len(spec.sources)} source(s); nothing was imported)"
            )
            return 0
        core = await acompile_spec(spec, allow_import=False if a.no_import_sources else None)
    except SpecError as e:
        return _fail(str(e))
    if a.sub == "inspect":
        print(json.dumps(describe(core), indent=2, ensure_ascii=False))
        return 0
    if a.sub == "check":
        print("ok: configuration is valid")
        for line in core.report:
            print(f"  {line}")
        return 0
    for line in core.report:  # stderr: stdout is the protocol
        print(f"aaa mcp: {line}", file=sys.stderr)
    if core.owns_engine and core.engine is not None:
        async with (
            core.engine
        ):  # the engine this configuration built runs for as long as the server does
            await serve_stdio(core)
    else:
        await serve_stdio(core)
    return 0


def main_run(a: argparse.Namespace) -> int:
    return asyncio.run(run(a))
