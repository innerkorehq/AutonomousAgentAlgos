"""Compiling an `MCPSpec` into a running server core.

    spec = load_spec("server.yaml")
    core = await acompile_spec(spec)          # or compile_spec(spec) outside an event loop
    await serve_stdio(core)                   # or ServerConfig(mcp=core, ...) behind aaa-server

Importing code is the one dangerous thing a spec can ask for, so it has a kill-switch: `AAA_MCP_ALLOW_IMPORT=0` (or
`allow_import=False`) turns any source that imports a module into an error before anything is imported."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from aaa.mcp.config import SpecError
from aaa.mcp.filters import Entry, rename, select
from aaa.mcp.model import MCPAnnotations, MCPToolMeta
from aaa.mcp.server import ApprovalCallback, MCPToolServer
from aaa.mcp.sources.objects import load_ref, normalize
from aaa.mcp.spec import AnnotationsSpec, MCPSpec, ToolOverride
from aaa.tools.tool import ToolRegistry

if TYPE_CHECKING:
    from aaa.runtime.engine import Engine


def import_allowed() -> bool:
    return os.environ.get("AAA_MCP_ALLOW_IMPORT", "1").strip().lower() not in ("0", "false", "no")


class CompiledServer(MCPToolServer):
    """An `MCPToolServer` built from a spec, plus what was decided: the spec itself and a human-readable `report`."""

    spec: MCPSpec
    report: list[str]
    hidden: list[str]


def _annotations(a: AnnotationsSpec | None) -> MCPAnnotations | None:
    if a is None:
        return None
    return MCPAnnotations(a.title, a.read_only, a.destructive, a.idempotent, a.open_world)


def _override(entry: Entry, o: ToolOverride) -> Entry:
    import copy

    tool = copy.copy(entry.tool)
    if o.description is not None:
        tool.description = o.description
    if o.requires_approval is not None:
        tool.requires_approval = o.requires_approval
    return Entry(tool, entry.origin, entry.tags)


async def acompile_spec(
    spec: MCPSpec,
    *,
    live: Mapping[str, Any] | None = None,
    engine: Engine | None = None,
    approval: ApprovalCallback | None = None,
    allow_import: bool | None = None,
) -> CompiledServer:
    allowed = import_allowed() if allow_import is None else allow_import
    if spec.imports_code() and not allowed:
        refs = [s.ref for s in spec.sources if s.type == "objects"]
        raise SpecError(
            f"sources {refs} import code, but importing is disabled (AAA_MCP_ALLOW_IMPORT=0 / --no-import-sources)"
        )
    entries: list[Entry] = []
    for src in spec.sources:
        if src.type == "objects":
            tools, origin = await load_ref(src.ref), f"objects {src.ref}"
        else:
            if live is None or src.key not in live:
                raise SpecError(
                    f"sources: no live object was given for key {src.key!r} (compile_spec(..., live={{...}}))"
                )
            tools, origin = normalize(live[src.key], f"live {src.key}"), f"live {src.key}"
        from aaa.mcp.filters import with_name

        for t in tools:
            entries.append(Entry(with_name(t, src.prefix + t.name), origin, frozenset(src.tags)))

    chosen = rename(select(entries, spec.tools), spec.tools)
    unknown = set(spec.tools.overrides) - {e.tool.name for e in chosen}
    if unknown:
        raise SpecError(
            f"tools.overrides: no exposed tool is named {sorted(unknown)} (overrides use the final exposed name)"
        )
    meta: dict[str, MCPToolMeta] = {}
    final: list[Entry] = []
    for e in chosen:
        o = spec.tools.overrides.get(e.tool.name)
        if o is not None:
            e = _override(e, o)
            meta[e.tool.name] = MCPToolMeta(o.title, _annotations(o.annotations), o.output_schema)
        final.append(e)
    if len(final) > spec.limits.max_tools:
        raise SpecError(
            f"{len(final)} tools selected but limits.max_tools is {spec.limits.max_tools}: "
            "narrow tools.include/exclude/tags or raise the limit deliberately"
        )
    registry = ToolRegistry([e.tool for e in final])
    core = CompiledServer(
        registry,
        name=spec.server.name,
        title=spec.server.title,
        version=spec.server.version,
        instructions=spec.server.instructions,
        engine=engine,
        approval=approval,
        expose_approval_tools=spec.tools.expose_approval_tools and approval is None,
        tenant=spec.tenancy.tenant,
        user=spec.tenancy.user,
        max_result_chars=spec.limits.max_result_chars,
        tool_meta=meta,
        structured_output=spec.limits.structured_output,
    )
    core.spec = spec
    offered = {t.name for t in core.list_tools()}
    core.hidden = [e.tool.name for e in final if e.tool.name not in offered]
    core.report = [f"{len(offered)} tool(s) exposed from {len(spec.sources)} source(s)"]
    if core.hidden:
        core.report.append(
            f"{len(core.hidden)} tool(s) hidden because they require approval "
            f"(tools.expose_approval_tools, or an override with requires_approval: false): {', '.join(core.hidden)}"
        )
    if not offered:
        core.report.append("WARNING: the server exposes no tools")
    return core


def compile_spec(spec: MCPSpec, **kw: Any) -> CompiledServer:
    """`acompile_spec` for code that is not already in an event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(acompile_spec(spec, **kw))
    raise RuntimeError(
        "compile_spec() was called inside a running event loop: use `await acompile_spec(...)`"
    )


def describe(core: CompiledServer) -> dict[str, Any]:
    """Exactly what a client (and so a model) would be shown, for review: tool names, titles, descriptions and hints."""
    from aaa.mcp import wire

    return {
        "server": wire.server_info(core)
        | ({"instructions": core.instructions} if core.instructions else {}),
        "tools": wire.tools_list_result(core)["tools"],
        "hidden": core.hidden,
        "report": core.report,
    }


__all__ = ["CompiledServer", "acompile_spec", "compile_spec", "describe", "import_allowed"]
