"""The canonical MCP wire shapes (JSON-RPC result bodies) that every serializer builds on.

`aaa-server`'s stateless HTTP handler returns these dictionaries as they are; the stdio server builds the SDK's own types from them
(`Tool.model_validate(tool_to_wire(d))`), so a field cannot exist in one transport and be forgotten in another."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from aaa.mcp.model import (
    MCPAnnotations,
    MCPCallResult,
    MCPPrompt,
    MCPPromptResult,
    MCPResource,
    MCPResourceContents,
    MCPResourceTemplate,
    MCPToolDef,
)

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
    caps: dict[str, Any] = {"tools": {"listChanged": False}}
    if core.has_resources:
        caps["resources"] = {"subscribe": False, "listChanged": False}
    if core.has_prompts:
        caps["prompts"] = {"listChanged": False}
    return caps


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


def _drop_none(d: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if v is not None}


def resource_to_wire(r: MCPResource) -> dict[str, Any]:
    return _drop_none(
        {
            "uri": r.uri,
            "name": r.name,
            "title": r.title,
            "description": r.description,
            "mimeType": r.mime_type,
            "size": r.size,
        }
    )


def resource_template_to_wire(t: MCPResourceTemplate) -> dict[str, Any]:
    return _drop_none(
        {
            "uriTemplate": t.uri_template,
            "name": t.name,
            "title": t.title,
            "description": t.description,
            "mimeType": t.mime_type,
        }
    )


def resources_list_result(resources: list[MCPResource]) -> dict[str, Any]:
    return {"resources": [resource_to_wire(r) for r in resources]}


def resource_templates_list_result(templates: list[MCPResourceTemplate]) -> dict[str, Any]:
    return {"resourceTemplates": [resource_template_to_wire(t) for t in templates]}


def resource_read_result(c: MCPResourceContents) -> dict[str, Any]:
    return {"contents": [{"uri": c.uri, "mimeType": c.mime_type, "text": c.text}]}


def prompt_to_wire(p: MCPPrompt) -> dict[str, Any]:
    out = _drop_none({"name": p.name, "title": p.title, "description": p.description})
    out["arguments"] = [
        _drop_none({"name": a.name, "description": a.description, "required": a.required})
        for a in p.arguments
    ]
    return out


def prompts_list_result(prompts: list[MCPPrompt]) -> dict[str, Any]:
    return {"prompts": [prompt_to_wire(p) for p in prompts]}


def prompt_get_result(r: MCPPromptResult) -> dict[str, Any]:
    out: dict[str, Any] = {
        "messages": [
            {"role": role, "content": {"type": "text", "text": text}} for role, text in r.messages
        ]
    }
    if r.description:
        out["description"] = r.description
    return out
