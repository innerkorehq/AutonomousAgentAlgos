"""`MCPSpec`: the one description of a generated MCP server.

A YAML/JSON/TOML file, the Python builder, command-line flags and environment variables all produce an `MCPSpec`; the compiler
(`aaa.mcp.compile`) is the only thing that reads one. Unknown keys are errors, so a typo cannot silently do nothing."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

TOOL_NAME = r"^[A-Za-z0-9_-]{1,64}$"


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AnnotationsSpec(_M):
    """MCP tool hints: for clients to decide how cautious to be; the server enforces none of them."""

    title: str | None = Field(default=None, max_length=200)
    read_only: bool | None = None
    destructive: bool | None = None
    idempotent: bool | None = None
    open_world: bool | None = None


class ServerSpec(_M):
    name: str = Field(default="aaa", pattern=TOOL_NAME)
    title: str | None = Field(default=None, max_length=200)
    version: str = Field(default="0.1", max_length=40)
    instructions: str | None = Field(default=None, max_length=4000)


class TransportSpec(_M):
    """What `aaa mcp serve` does. Behind `aaa-server` the HTTP listener is the server's own, and these are ignored."""

    kind: Literal["stdio"] = "stdio"


class LimitsSpec(_M):
    max_result_chars: int = Field(default=100_000, ge=1_000, le=5_000_000)
    max_tools: int = Field(default=200, ge=1, le=2_000)
    structured_output: bool = (
        True  # send structuredContent for tools whose return type is an object type
    )


class TenancySpec(_M):
    """Who the calls are for on a stdio server. Behind `aaa-server` the authenticated principal replaces both."""

    tenant: str = Field(default="mcp", min_length=1, max_length=100, pattern=r"^[^/\s]+$")
    user: str | None = Field(default=None, max_length=200)


class ToolOverride(_M):
    title: str | None = Field(default=None, max_length=200)
    description: str | None = Field(default=None, max_length=2000)
    annotations: AnnotationsSpec | None = None
    requires_approval: bool | None = None
    output_schema: dict[str, Any] | None = None


class ToolsSpec(_M):
    """Which tools are exposed and how they look.

    `include`/`exclude`/`tags` match the name a source gave the tool (after the source's own `prefix`); `rename` keys are those names;
    the global `prefix` is added after renaming; `overrides` are keyed by the name the tool is finally exposed under.
    A tool that requires approval is hidden unless `expose_approval_tools` is true or its override sets `requires_approval: false`."""

    include: list[str] = Field(default_factory=lambda: ["*"])
    exclude: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    rename: dict[str, str] = Field(default_factory=dict)
    prefix: str = Field(default="", max_length=40, pattern=r"^[A-Za-z0-9_-]*$")
    overrides: dict[str, ToolOverride] = Field(default_factory=dict)
    expose_approval_tools: bool = False

    @field_validator("rename")
    @classmethod
    def _names(cls, v: dict[str, str]) -> dict[str, str]:
        import re

        for new in v.values():
            if not re.match(TOOL_NAME, new):
                raise ValueError(
                    f"{new!r} is not a valid tool name (letters, digits, '_' and '-', at most 64)"
                )
        return v


class _Source(_M):
    tags: list[str] = Field(default_factory=list)
    prefix: str = Field(default="", max_length=40, pattern=r"^[A-Za-z0-9_-]*$")


class ObjectsSource(_Source):
    """Tools from AAA objects defined in your code: `ref` is `package.module:attribute`, and the attribute is a `ToolRegistry`, an
    `MCPToolServer`, a `Tool`, a list of tools, or a function (sync or async) returning one of those."""

    type: Literal["objects"]
    ref: str = Field(pattern=r"^[A-Za-z_][\w.]*:[A-Za-z_][\w.]*$")


class LiveSource(_Source):
    """Tools passed to `compile_spec(..., live={key: ...})` from Python. Not loadable from a file (there is nothing to load)."""

    type: Literal["live"]
    key: str = Field(min_length=1, max_length=80)


Source = Annotated[ObjectsSource | LiveSource, Field(discriminator="type")]


class MCPSpec(_M):
    version: Literal[1] = 1
    server: ServerSpec = Field(default_factory=ServerSpec)
    transport: TransportSpec = Field(default_factory=TransportSpec)
    limits: LimitsSpec = Field(default_factory=LimitsSpec)
    tenancy: TenancySpec = Field(default_factory=TenancySpec)
    sources: list[Source] = Field(default_factory=list)
    tools: ToolsSpec = Field(default_factory=ToolsSpec)

    def imports_code(self) -> bool:
        """Whether compiling this spec imports modules (and so runs code): see the kill-switch in `aaa.mcp.compile`."""
        return any(s.type == "objects" for s in self.sources)


__all__ = [
    "AnnotationsSpec",
    "LimitsSpec",
    "LiveSource",
    "MCPSpec",
    "ObjectsSource",
    "ServerSpec",
    "Source",
    "TenancySpec",
    "ToolOverride",
    "ToolsSpec",
    "TransportSpec",
]
