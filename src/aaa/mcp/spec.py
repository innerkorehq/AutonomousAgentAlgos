"""`MCPSpec`: the one description of a generated MCP server.

A YAML/JSON/TOML file, the Python builder, command-line flags and environment variables all produce an `MCPSpec`; the compiler
(`aaa.mcp.compile`) is the only thing that reads one. Unknown keys are errors, so a typo cannot silently do nothing."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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
    max_concurrent_agents: int = Field(
        default=4, ge=1, le=64
    )  # agent runs in flight across all agent tools
    max_resource_bytes: int = Field(
        default=262_144, ge=1_000, le=10_000_000
    )  # one resource read is cut here
    max_resources_listed: int = Field(default=500, ge=1, le=10_000)
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


class BudgetSpec(_M):
    """What one agent run may spend. `deadline_s` is mandatory: a run that parks on a human approval (which MCP cannot give)
    would otherwise wait forever."""

    deadline_s: float = Field(gt=0, le=3600)
    max_steps: int | None = Field(default=None, ge=1)
    max_tokens: int | None = Field(default=None, ge=1)
    max_cost: float | None = Field(default=None, gt=0)


DEFAULT_AGENT_INPUT: dict[str, Any] = {
    "type": "object",
    "properties": {"input": {"type": "string", "description": "What to ask the agent."}},
    "required": ["input"],
    "additionalProperties": False,
}


class AgentToolSpec(_M):
    """Expose one algorithm as its own tool. `algorithm` is a `module:attribute` reference (an `Algorithm`) or the key of an object
    passed to `compile_spec(..., live={...})`. `input_template` builds the algorithm's input from the validated arguments:
    string values may contain `{argument}` placeholders (plain names only; no attribute access, no format specs) and a value that is
    exactly one placeholder keeps the argument's own type."""

    name: str = Field(pattern=TOOL_NAME)
    algorithm: str = Field(min_length=1, max_length=200)
    description: str = Field(min_length=1, max_length=2000)
    budget: BudgetSpec
    input_schema: dict[str, Any] = Field(default_factory=lambda: dict(DEFAULT_AGENT_INPUT))
    input_template: dict[str, Any] = Field(default_factory=lambda: {"task": "{input}"})
    output: Literal["text", "json"] = "text"  # the answer's text, or the whole outcome as JSON
    annotations: AnnotationsSpec | None = None
    tags: list[str] = Field(default_factory=list)


class RuntimeSpec(_M):
    """The engine agent tools run on, for a server that builds its own (`aaa mcp serve`). `engine` is `module:attribute`: an
    `Engine` or a function (sync or async) returning one, already wired with your model and tool handlers. Behind
    `aaa-server` the server's engine is used and this is ignored."""

    engine: str | None = Field(default=None, pattern=r"^[A-Za-z_][\w.]*:[A-Za-z_][\w.]*$")


class StaticResource(_M):
    type: Literal["static"]
    name: str = Field(pattern=TOOL_NAME)
    text: str = Field(max_length=200_000)
    mime_type: str = "text/plain"
    title: str | None = Field(default=None, max_length=200)
    description: str | None = Field(default=None, max_length=1000)


class FilesResource(_M):
    """Files under `root` as resources. Per-tenant by default: a caller sees only `root/<tenant>/...`; `shared = true` serves `root`
    itself to every tenant and must be said explicitly. Hidden files and key-like names are never served."""

    type: Literal["files"]
    name: str = Field(pattern=TOOL_NAME)
    root: str = Field(min_length=1)
    pattern: str = "**/*"
    shared: bool = False
    allow_hidden: bool = False
    mime_type: str | None = None
    title: str | None = Field(default=None, max_length=200)
    description: str | None = Field(default=None, max_length=1000)


class IndexResource(_M):
    """Search over an `Index` (`ref` = `module:attribute`) as `aaa://index/<name>/search?q=...`. Documents are tenant-scoped through
    their `meta[tenant_key]`; an index holding one tenant's or public data says `shared = true` instead."""

    type: Literal["index"]
    name: str = Field(pattern=TOOL_NAME)
    ref: str = Field(min_length=1, max_length=200)  # module:attribute, or the key of a live object
    k: int = Field(default=5, ge=1, le=20)
    max_chars: int = Field(default=800, ge=50, le=20_000)
    tenant_key: str | None = None
    shared: bool = False
    title: str | None = Field(default=None, max_length=200)
    description: str | None = Field(default=None, max_length=1000)

    @model_validator(mode="after")
    def _scoped(self) -> IndexResource:
        if not self.shared and not self.tenant_key:
            raise ValueError(
                "set tenant_key (the document meta field holding the tenant) or shared = true"
            )
        return self


class MemoryResource(_M):
    """Search over a memory backend (`ref` = `module:attribute`), scoped by the caller's tenant and user."""

    type: Literal["memory"]
    name: str = Field(pattern=TOOL_NAME)
    ref: str = Field(min_length=1, max_length=200)  # module:attribute, or the key of a live object
    k: int = Field(default=5, ge=1, le=20)
    max_chars: int = Field(default=800, ge=50, le=20_000)
    title: str | None = Field(default=None, max_length=200)
    description: str | None = Field(default=None, max_length=1000)


ResourceSpec = Annotated[
    StaticResource | FilesResource | IndexResource | MemoryResource, Field(discriminator="type")
]


class PromptArg(_M):
    name: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$", max_length=64)
    description: str | None = Field(default=None, max_length=500)
    required: bool = True
    type: Literal["string", "integer", "boolean", "enum"] = "string"
    choices: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _choices(self) -> PromptArg:
        if (self.type == "enum") != bool(self.choices):
            raise ValueError("choices are required for, and only for, type = 'enum'")
        return self


class PromptMessage(_M):
    role: Literal["user", "assistant"] = "user"
    text: str = Field(min_length=1, max_length=20_000)


class PromptSpec(_M):
    """A prompt template. `{argument}` placeholders in message texts are filled from the (validated) arguments; write a literal brace
    as `{{` or `}}`."""

    name: str = Field(pattern=TOOL_NAME)
    title: str | None = Field(default=None, max_length=200)
    description: str | None = Field(default=None, max_length=1000)
    arguments: list[PromptArg] = Field(default_factory=list)
    messages: list[PromptMessage] = Field(min_length=1)

    @model_validator(mode="after")
    def _placeholders(self) -> PromptSpec:
        from aaa.mcp.config import SpecError
        from aaa.mcp.templating import template_fields

        names = [a.name for a in self.arguments]
        if len(set(names)) != len(names):
            raise ValueError("duplicate argument names")
        try:
            used = set().union(
                *(template_fields(m.text, where=f"prompts.{self.name}") for m in self.messages)
            )
        except SpecError as e:
            raise ValueError(str(e)) from None
        if missing := sorted(used - set(names)):
            raise ValueError(f"message placeholders {missing} are not declared as arguments")
        return self


class _Source(_M):
    tags: list[str] = Field(default_factory=list)
    prefix: str = Field(default="", max_length=40, pattern=r"^[A-Za-z0-9_-]*$")


class ObjectsSource(_Source):
    """Tools from AAA objects defined in your code: `ref` is `package.module:attribute`, and the attribute is a `ToolRegistry`, an
    `MCPToolServer`, a `Tool`, a list of tools, or a function (sync or async) returning one of those."""

    type: Literal["objects"]
    ref: str = Field(pattern=r"^[A-Za-z_][\w.]*:[A-Za-z_][\w.]*$")


class ModuleSource(_Source):
    """Public functions of a Python module as tools: schema from type hints, description from the docstring.

    Importing a module runs its top-level code, so this is as trusted as `objects`, and the list of what is exposed is explicit:
    `allow` names functions, or the module's `__all__` is used; a module with neither is an error. Only functions DEFINED in the
    module are exposed (re-exports need `allow`), never `_private` names, classes, generators, lambdas or `*args/**kwargs`.
    Nothing is ever called while the module is inspected.

    Module tools REQUIRE APPROVAL by default (so they stay hidden): list the harmless ones in `no_approval`, or in `read_only`
    (which also tells clients they only read)."""

    type: Literal["module"]
    module: str = Field(pattern=r"^[A-Za-z_][\w]*(\.[A-Za-z_][\w]*)*$", max_length=200)
    allow: list[str] = Field(default_factory=list)
    exclude: list[str] = Field(default_factory=list)
    path: str | None = (
        None  # a directory to add to sys.path so the module can be found (explicit, never guessed)
    )
    strict: bool = True  # every parameter needs a type hint; false types the unannotated ones as Any, with a warning
    default_approval: bool = True
    no_approval: list[str] = Field(default_factory=list)
    read_only: list[str] = Field(default_factory=list)


class LiveSource(_Source):
    """Tools passed to `compile_spec(..., live={key: ...})` from Python. Not loadable from a file (there is nothing to load)."""

    type: Literal["live"]
    key: str = Field(min_length=1, max_length=80)


Source = Annotated[ObjectsSource | ModuleSource | LiveSource, Field(discriminator="type")]


class MCPSpec(_M):
    version: Literal[1] = 1
    server: ServerSpec = Field(default_factory=ServerSpec)
    transport: TransportSpec = Field(default_factory=TransportSpec)
    limits: LimitsSpec = Field(default_factory=LimitsSpec)
    tenancy: TenancySpec = Field(default_factory=TenancySpec)
    sources: list[Source] = Field(default_factory=list)
    tools: ToolsSpec = Field(default_factory=ToolsSpec)
    agents: list[AgentToolSpec] = Field(default_factory=list)
    resources: list[ResourceSpec] = Field(default_factory=list)
    prompts: list[PromptSpec] = Field(default_factory=list)
    runtime: RuntimeSpec = Field(default_factory=RuntimeSpec)

    def imports_code(self) -> bool:
        """Whether compiling this spec imports modules (and so runs code): see the kill-switch in `aaa.mcp.compile`."""
        return (
            any(s.type in ("objects", "module") for s in self.sources)
            or self.runtime.engine is not None
            or any(":" in a.algorithm for a in self.agents)
            or any(
                isinstance(r, IndexResource | MemoryResource) and ":" in r.ref
                for r in self.resources
            )
        )


__all__ = [
    "AgentToolSpec",
    "AnnotationsSpec",
    "BudgetSpec",
    "FilesResource",
    "IndexResource",
    "LimitsSpec",
    "LiveSource",
    "MCPSpec",
    "MemoryResource",
    "ModuleSource",
    "ObjectsSource",
    "PromptArg",
    "PromptMessage",
    "PromptSpec",
    "ResourceSpec",
    "RuntimeSpec",
    "ServerSpec",
    "Source",
    "StaticResource",
    "TenancySpec",
    "ToolOverride",
    "ToolsSpec",
    "TransportSpec",
]
