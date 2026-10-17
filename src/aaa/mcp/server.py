# pyright: reportMissingImports=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportArgumentType=false
# (the `mcp` SDK is an optional dependency with two incompatible major versions)
"""Expose AAA tools (and, optionally, whole algorithms) as an MCP server.

    server = MCPToolServer(registry)                         # tools only
    server = MCPToolServer(registry, engine=engine, algorithms={"react": ReAct(registry)})
    await serve_stdio(server)                                # or: aaa mcp serve mymodule:build

The defaults assume the caller is not the person who wrote the tools:

* **Tools that need human approval are not listed and cannot be called** (a call is answered exactly like a call to a
  tool that does not exist). MCP has no approval channel, so the only ways to expose one are an explicit
  `approval=` callback that makes the decision, or `expose_approval_tools=True`, which says in its name that you accept the risk.
  Nothing is ever approved automatically.
* **Arguments are validated by the tool's own schema** before it runs; failures and exceptions become `isError` results with a
  short message (a tool's `ToolError` text is shown, any other exception is logged and reported generically).
* **The tenant and user are fixed by whoever builds the server** (or, behind `aaa-server`, by the authenticated principal);
  nothing a client sends can change them.
* **`run_agent` (algorithms) is off unless you pass `algorithms=`**, always runs under a `Budget` with a deadline, and returns
  the outcome only.

`MCPToolServer` is the SDK-free core (testable on its own); `build_server` and `serve_stdio` bind it to the official `mcp` SDK
(1.x or 2.x)."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import Awaitable, Callable, Mapping
from typing import TYPE_CHECKING, Any

from aaa.mcp.model import MCPAnnotations, MCPCallResult, MCPToolDef, MCPToolMeta
from aaa.runtime.budget import Budget
from aaa.tools.tool import Tool, ToolArgsError, ToolContext, ToolError, ToolRegistry

if TYPE_CHECKING:
    from aaa.runtime.engine import Algorithm, Engine

log = logging.getLogger("aaa.mcp.server")

RUN_AGENT = "run_agent"
ApprovalCallback = Callable[[str, dict[str, Any], str | None], Awaitable[bool]]
"""`(tool name, arguments, tenant) -> allowed`. Called for every call to a tool that requires approval."""


class MCPToolServer:
    def __init__(
        self,
        registry: ToolRegistry,
        *,
        name: str = "aaa",
        engine: Engine | None = None,
        algorithms: Mapping[str, Algorithm] | None = None,
        approval: ApprovalCallback | None = None,
        expose_approval_tools: bool = False,
        tenant: str = "mcp",
        user: str | None = None,
        agent_budget: Budget | None = None,
        max_result_chars: int = 100_000,
        title: str | None = None,
        version: str = "0.1",
        instructions: str | None = None,
        tool_meta: Mapping[str, MCPToolMeta] | None = None,
        structured_output: bool = False,
    ) -> None:
        if algorithms and engine is None:
            raise ValueError("algorithms= needs engine=")
        if expose_approval_tools and approval is not None:
            raise ValueError("pass either approval= or expose_approval_tools=True, not both")
        self.registry, self.name, self.engine = registry, name, engine
        self.algorithms = dict(algorithms or {})
        self.approval, self.expose_approval_tools = approval, expose_approval_tools
        self.tenant, self.user, self.max_result_chars = tenant, user, max_result_chars
        self.title, self.version, self.instructions = title, version, instructions
        self.tool_meta = dict(tool_meta or {})
        self.structured_output = (
            structured_output  # send structuredContent for tools with an output schema
        )
        self.agent_budget = agent_budget or Budget(
            max_steps=50, max_tokens=200_000, deadline_s=120.0
        )

    # -- what is offered -------------------------------------------------------------------
    def _exposed(self, t: Tool) -> bool:
        return (not t.requires_approval) or self.approval is not None or self.expose_approval_tools

    def list_tools(self) -> list[MCPToolDef]:
        out = [
            self._tool_def(t)
            for n in self.registry.names()
            if (t := self.registry.get(n)) is not None and self._exposed(t)
        ]
        if self.algorithms:
            out.append(
                MCPToolDef(
                    RUN_AGENT,
                    "Run an agent algorithm to completion and return its outcome.",
                    {
                        "type": "object",
                        "properties": {
                            "algo": {"type": "string", "enum": sorted(self.algorithms)},
                            "input": {"type": "object"},
                        },
                        "required": ["algo"],
                        "additionalProperties": False,
                    },
                )
            )
        return out

    def _output_schema(self, t: Tool) -> dict[str, Any] | None:
        meta = self.tool_meta.get(t.name)
        if meta is not None and meta.output_schema is not None:
            return meta.output_schema
        return t.output_schema if self.structured_output else None

    def _tool_def(self, t: Tool) -> MCPToolDef:
        meta = self.tool_meta.get(t.name) or MCPToolMeta()
        ann = meta.annotations or _default_annotations(t)
        return MCPToolDef(
            t.name,
            t.description,
            t.parameters,
            read_only=bool(ann and ann.read_only),
            title=meta.title or (ann.title if ann else None),
            annotations=ann,
            output_schema=self._output_schema(t),
        )

    # -- calling ---------------------------------------------------------------------------
    def _render(self, value: Any) -> str:
        text = (
            value if isinstance(value, str) else json.dumps(value, default=str, ensure_ascii=False)
        )
        if len(text) > self.max_result_chars:
            text = (
                text[: self.max_result_chars]
                + f"\n[truncated at {self.max_result_chars} characters]"
            )
        return text

    async def call(
        self,
        name: str,
        arguments: Mapping[str, Any] | None = None,
        *,
        tenant: str | None = None,
        user: str | None = None,
    ) -> MCPCallResult:
        """Run one tool. `tenant`/`user` are for a host that has authenticated its caller (aaa-server); the
        MCP client never supplies them."""
        args = dict(arguments or {})
        tenant, user = tenant or self.tenant, user or self.user
        if name == RUN_AGENT and self.algorithms:
            return await self._run_agent(args, tenant, user)
        t = self.registry.get(name)
        if t is None or not self._exposed(t):
            return MCPCallResult(f"unknown tool {name!r}", True)
        if t.requires_approval and self.approval is not None:
            try:
                allowed = await self.approval(name, args, tenant)
            except Exception:
                log.exception("approval callback failed for %s", name)
                allowed = False
            if not allowed:
                return MCPCallResult(f"{name} was not approved", True)
        ctx = ToolContext(
            engine=self.engine,
            run_id=f"mcp:{uuid.uuid4().hex}",
            call_id=uuid.uuid4().hex,
            tenant=tenant,
            meta={"user": user} if user else {},
        )
        try:
            async with asyncio.timeout(t.timeout_s):
                value = await t.invoke(args, ctx)
            return self._result(t, value)
        except TimeoutError:
            return MCPCallResult(f"{name} timed out after {t.timeout_s:g}s", True)
        except ToolArgsError as e:
            return MCPCallResult(str(e), True)
        except ToolError as e:
            return MCPCallResult(str(e) or f"{name} failed", True)
        except Exception:
            log.exception("tool %s failed", name)
            return MCPCallResult(f"{name} failed", True)

    def _result(self, t: Tool, value: Any) -> MCPCallResult:
        schema = self._output_schema(t)
        structured = None
        if schema is not None and isinstance(value, dict):
            problem = _violates(schema, value)
            if problem:
                log.error(
                    "tool %s returned data that does not match its output schema: %s",
                    t.name,
                    problem,
                )
                return MCPCallResult(
                    f"{t.name} returned data that does not match its output schema", True
                )
            structured = value
        text = self._render(value)
        if (
            structured is not None
            and len(json.dumps(structured, default=str)) > self.max_result_chars
        ):
            structured = None  # never send a structure that was cut short: text only, with its truncation marker
        return MCPCallResult(text, False, structured)

    async def _run_agent(
        self, args: dict[str, Any], tenant: str, user: str | None
    ) -> MCPCallResult:
        assert self.engine is not None
        extra = set(args) - {"algo", "input"}
        algo = self.algorithms.get(str(args.get("algo")))
        inp = args.get("input", {})
        if extra or algo is None or not isinstance(inp, dict):
            return MCPCallResult("run_agent needs {algo: <one of the offered>, input: {...}}", True)
        try:
            out = await self.engine.tenant(tenant).run(
                algo, inp, budget=self.agent_budget, user=user
            )
        except Exception:
            log.exception("run_agent failed")
            return MCPCallResult("the agent run failed", True)
        body = {"kind": out.kind.value, "value": out.value, "reason": out.reason}
        return MCPCallResult(self._render(body), not out.ok)


def _default_annotations(t: Tool) -> MCPAnnotations | None:
    """Hints derivable from what the tool already says: approval-gated means potentially destructive, and a tool whose delivery
    is `at_least_once` is safe to repeat (idempotent). Nothing is claimed about read-only-ness: only a config can say so."""
    destructive = True if t.requires_approval else None
    idempotent = True if t.delivery == "at_least_once" else None
    if destructive is None and idempotent is None:
        return None
    return MCPAnnotations(destructive=destructive, idempotent=idempotent)


def _violates(schema: dict[str, Any], value: Any) -> str | None:
    try:
        import jsonschema  # pyright: ignore[reportMissingModuleSource]
    except (
        ImportError
    ):  # the mcp extra brings it; without it the value is trusted as the tool's own
        return None
    try:
        jsonschema.validate(value, schema)
    except jsonschema.ValidationError as e:
        return e.message[:200]
    except jsonschema.SchemaError as e:
        return f"invalid output schema: {e.message[:120]}"
    return None


# -- binding to the official SDK ---------------------------------------------------------------
def build_server(core: MCPToolServer) -> Any:
    """An SDK server object (low-level `Server`) exposing `core`. Works with mcp 1.x and 2.x.

    The SDK's own types are built from the shared wire dictionaries (`aaa.mcp.wire`), so what a tool looks like is decided in one
    place for stdio, the official client and the HTTP handler alike."""
    import mcp

    from aaa.mcp import wire

    if hasattr(
        mcp, "Client"
    ):  # mcp >= 2 (not "can mcp_types be imported": it can outlive a downgrade to 1.x)
        import mcp_types as types
        from mcp.server.lowlevel import Server

        async def on_list(ctx: Any, params: Any) -> Any:
            return types.ListToolsResult.model_validate(wire.tools_list_result(core))

        async def on_call(ctx: Any, params: Any) -> Any:
            r = await core.call(params.name, params.arguments)
            return types.CallToolResult.model_validate(wire.call_result_to_wire(r))

        return Server(
            core.name,
            version=core.version,
            title=core.title,
            instructions=core.instructions,
            on_list_tools=on_list,
            on_call_tool=on_call,
        )
    from mcp import types  # mcp 1.x
    from mcp.server.lowlevel import Server

    server = Server(core.name, version=core.version, instructions=core.instructions)

    @server.list_tools()
    async def _list() -> list[Any]:
        return [types.Tool.model_validate(w) for w in wire.tools_list_result(core)["tools"]]

    @server.call_tool(validate_input=False)
    async def _call(name: str, arguments: dict[str, Any] | None) -> Any:
        r = await core.call(name, arguments)
        if r.is_error:
            raise ToolError(
                r.text
            )  # the SDK turns an exception into an isError result carrying this text
        content = [types.TextContent(type="text", text=r.text)]
        return (content, r.structured) if r.structured is not None else content

    return server


async def serve_stdio(core: MCPToolServer) -> None:
    """Serve over stdio until the client disconnects (stdout carries the protocol: never print to it)."""
    from mcp.server.stdio import stdio_server

    server = build_server(core)
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


__all__ = [
    "RUN_AGENT",
    "ApprovalCallback",
    "MCPAnnotations",
    "MCPCallResult",
    "MCPToolDef",
    "MCPToolMeta",
    "MCPToolServer",
    "build_server",
    "serve_stdio",
]
