"""The SDK-free vocabulary of the MCP server: what is offered and what a call returns.

These dataclasses are wire-neutral. `aaa.mcp.wire` turns them into the JSON-RPC dictionaries that the SDK adapters and the
HTTP handler of `aaa-server` all share, so the three serializers cannot disagree about what a tool looks like."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MCPAnnotations:
    """MCP tool hints. They are hints for clients, not guarantees: nothing here is enforced by the server."""

    title: str | None = None
    read_only: bool | None = None
    destructive: bool | None = None
    idempotent: bool | None = None
    open_world: bool | None = None


@dataclass(frozen=True)
class MCPToolMeta:
    """Per-tool presentation overrides (what a config's `tools.overrides` produces)."""

    title: str | None = None
    annotations: MCPAnnotations | None = None
    output_schema: dict[str, Any] | None = None


@dataclass(frozen=True)
class MCPToolDef:
    name: str
    description: str
    input_schema: dict[str, Any]
    read_only: bool = False
    title: str | None = None
    annotations: MCPAnnotations | None = None
    output_schema: dict[str, Any] | None = None


@dataclass(frozen=True)
class MCPCallResult:
    text: str
    is_error: bool = False
    structured: dict[str, Any] | None = (
        None  # set only for a tool with an output schema that the result satisfied
    )


__all__ = ["MCPAnnotations", "MCPCallResult", "MCPToolDef", "MCPToolMeta"]
