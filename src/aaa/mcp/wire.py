"""The canonical MCP wire shapes (JSON-RPC result bodies) that every serializer builds on.

`aaa-server`'s stateless HTTP handler returns these dictionaries as they are; the stdio server builds the SDK's own types from them
(`Tool.model_validate(tool_to_wire(d))`), so a field cannot exist in one transport and be forgotten in another."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from aaa.mcp.model import MCPAnnotations, MCPCallResult, MCPToolDef

if TYPE_CHECKING:
    from aaa.mcp.server import MCPToolServer

_HINTS = (
    ("title", "title"),
    ("read_only", "readOnlyHint"),
    ("destructive", "destructiveHint"),
    ("idempotent", "idempotentHint"),
    ("open_world", "openWorldHint"),
)


def annotations_to_wire(a: MCPAnnotations | None) -> dict[str, Any] | None:
    if a is None:
        return None
    out = {wire: getattr(a, attr) for attr, wire in _HINTS if getattr(a, attr) is not None}
    return out or None


def tool_to_wire(d: MCPToolDef) -> dict[str, Any]:
    out: dict[str, Any] = {
        "name": d.name,
        "description": d.description,
        "inputSchema": d.input_schema,
    }
    if d.title:
        out["title"] = d.title
    if d.output_schema:
        out["outputSchema"] = d.output_schema
    if (ann := annotations_to_wire(d.annotations)) is not None:
        out["annotations"] = ann
    return out


def call_result_to_wire(r: MCPCallResult) -> dict[str, Any]:
    out: dict[str, Any] = {"content": [{"type": "text", "text": r.text}], "isError": r.is_error}
    if r.structured is not None and not r.is_error:
        out["structuredContent"] = r.structured
    return out


def capabilities(core: MCPToolServer) -> dict[str, Any]:
    return {"tools": {"listChanged": False}}


def server_info(core: MCPToolServer) -> dict[str, Any]:
    info: dict[str, Any] = {"name": core.name, "version": core.version}
    if core.title:
        info["title"] = core.title
    return info


def initialize_result(core: MCPToolServer, protocol_version: str) -> dict[str, Any]:
    out: dict[str, Any] = {
        "protocolVersion": protocol_version,
        "capabilities": capabilities(core),
        "serverInfo": server_info(core),
    }
    if core.instructions:
        out["instructions"] = core.instructions
    return out


def tools_list_result(core: MCPToolServer) -> dict[str, Any]:
    return {"tools": [tool_to_wire(t) for t in core.list_tools()]}
