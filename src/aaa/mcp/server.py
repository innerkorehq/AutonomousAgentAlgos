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

import json
import logging
import uuid
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from aaa.runtime.budget import Budget
from aaa.tools.tool import Tool, ToolArgsError, ToolContext, ToolError, ToolRegistry

if TYPE_CHECKING:
    from aaa.runtime.engine import Algorithm, Engine

log = logging.getLogger("aaa.mcp.server")

RUN_AGENT = "run_agent"
ApprovalCallback = Callable[[str, dict[str, Any], str | None], Awaitable[bool]]
"""`(tool name, arguments, tenant) -> allowed`. Called for every call to a tool that requires approval."""


@dataclass(frozen=True)
class MCPToolDef:
    name: str
    description: str
    input_schema: dict[str, Any]
    read_only: bool = False


@dataclass(frozen=True)
class MCPCallResult:
    text: str
    is_error: bool = False


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
    ) -> None:
        if algorithms and engine is None:
            raise ValueError("algorithms= needs engine=")
        if expose_approval_tools and approval is not None:
            raise ValueError("pass either approval= or expose_approval_tools=True, not both")
        self.registry, self.name, self.engine = registry, name, engine
        self.algorithms = dict(algorithms or {})
        self.approval, self.expose_approval_tools = approval, expose_approval_tools
        self.tenant, self.user, self.max_result_chars = tenant, user, max_result_chars
        self.agent_budget = agent_budget or Budget(
            max_steps=50, max_tokens=200_000, deadline_s=120.0
        )

    # -- what is offered -------------------------------------------------------------------
    def _exposed(self, t: Tool) -> bool:
        return (not t.requires_approval) or self.approval is not None or self.expose_approval_tools

    def list_tools(self) -> list[MCPToolDef]:
        out = [
            MCPToolDef(t.name, t.description, t.parameters)
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
            return MCPCallResult(self._render(await t.invoke(args, ctx)))
        except ToolArgsError as e:
            return MCPCallResult(str(e), True)
        except ToolError as e:
            return MCPCallResult(str(e) or f"{name} failed", True)
        except Exception:
            log.exception("tool %s failed", name)
            return MCPCallResult(f"{name} failed", True)

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


# -- binding to the official SDK ---------------------------------------------------------------
def build_server(core: MCPToolServer) -> Any:
    """An SDK server object (low-level `Server`) exposing `core`. Works with mcp 1.x and 2.x."""
    import mcp

    if hasattr(
        mcp, "Client"
    ):  # mcp >= 2 (not "can mcp_types be imported": it can outlive a downgrade to 1.x)
        import mcp_types as types
        from mcp.server.lowlevel import Server

        def tool_of(d: MCPToolDef) -> Any:
            return types.Tool(name=d.name, description=d.description, input_schema=d.input_schema)

        async def on_list(ctx: Any, params: Any) -> Any:
            return types.ListToolsResult(tools=[tool_of(d) for d in core.list_tools()])

        async def on_call(ctx: Any, params: Any) -> Any:
            r = await core.call(params.name, params.arguments)
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=r.text)], is_error=r.is_error
            )

        return Server(core.name, on_list_tools=on_list, on_call_tool=on_call)
    from mcp import types  # mcp 1.x
    from mcp.server.lowlevel import Server

    server = Server(core.name)

    @server.list_tools()
    async def _list() -> list[Any]:
        return [
            types.Tool(name=d.name, description=d.description, inputSchema=d.input_schema)
            for d in core.list_tools()
        ]

    @server.call_tool(validate_input=False)
    async def _call(name: str, arguments: dict[str, Any] | None) -> Any:
        r = await core.call(name, arguments)
        if r.is_error:
            raise ToolError(
                r.text
            )  # the SDK turns an exception into an isError result carrying this text
        return [types.TextContent(type="text", text=r.text)]

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
    "MCPCallResult",
    "MCPToolDef",
    "MCPToolServer",
    "build_server",
    "serve_stdio",
]
