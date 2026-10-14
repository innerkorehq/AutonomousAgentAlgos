# pyright: reportMissingImports=false, reportAttributeAccessIssue=false
# (the `mcp` SDK is an optional dependency with two incompatible major versions)
"""An MCP (Model Context Protocol) client that turns a server's tools into ordinary AAA tools.

    async with MCPClient.stdio("uvx", ["mcp-server-git"]) as git:
        registry = ToolRegistry(git.tools(prefix="git_"))
    async with MCPClient.http("https://mcp.example.com/mcp", headers={"Authorization": "Bearer ..."}) as srv:
        registry.extend(ToolRegistry(srv.tools(exclude=["delete_*"], approval="destructive")))

Those tools run through `ToolHandler` like any other, so they inherit argument validation, timeouts,
approval and the execution guard. What MCP adds is a trust problem, and the defaults are written for it:

* **Tool descriptions and results are untrusted input** (a server, or the data it relays, can try to steer
  the model). Filter with `include`/`exclude`, cap result size (`max_result_chars`), and keep sensitive
  tools behind `approval=`.
* **Annotations are hints, not guarantees.** `readOnlyHint`/`idempotentHint` let a tool be retried after a
  worker dies (`at_least_once`); everything else is `at_most_once`, and per the MCP spec a tool with no
  annotations counts as destructive. Pass `trust_annotations=False` for servers you do not control.
* **One connection per process.** In a cluster every worker opens its own (a stdio server is a subprocess).

Works with both major versions of the `mcp` SDK. Hand `MCPClient(session)` any already-open
`mcp.Client` (2.x) or `mcp.ClientSession` (1.x) to manage the connection yourself."""

from __future__ import annotations

import fnmatch
import hashlib
import inspect
import logging
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any, Literal, Self

from aaa.tools.tool import SchemaTool, Tool, ToolError

log = logging.getLogger("aaa.mcp")

Approval = Literal["none", "destructive", "always"]


def _attr(obj: Any, *names: str, default: Any = None) -> Any:
    """Read a field that the SDK spells `snake_case` in 2.x and `camelCase` in 1.x (or via a dict)."""
    for n in names:
        if isinstance(obj, Mapping):
            if n in obj:
                return obj[n]
        elif hasattr(obj, n):
            return getattr(obj, n)
    return default


def mcp_tool_name(prefix: str, name: str, *, max_len: int = 64) -> str:
    """A model-API-safe tool name: `[A-Za-z0-9_-]`, at most 64 characters, stable and unique."""
    raw = f"{prefix}{name}"
    clean = re.sub(r"[^A-Za-z0-9_-]", "_", raw)
    if len(clean) > max_len:
        clean = f"{clean[: max_len - 9]}_{hashlib.sha1(raw.encode()).hexdigest()[:8]}"
    return clean


@dataclass(frozen=True)
class MCPToolInfo:
    """A server tool, normalized across SDK versions."""

    name: str
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=lambda: {"type": "object", "properties": {}})
    title: str | None = None
    read_only: bool | None = None
    destructive: bool | None = None
    idempotent: bool | None = None
    open_world: bool | None = None

    @classmethod
    def of(cls, t: Any) -> MCPToolInfo:
        ann = _attr(t, "annotations")
        return cls(
            name=str(_attr(t, "name")),
            description=str(_attr(t, "description", default="") or ""),
            parameters=dict(
                _attr(t, "input_schema", "inputSchema", default=None)
                or {"type": "object", "properties": {}}
            ),
            title=_attr(t, "title"),
            read_only=_attr(ann, "read_only_hint", "readOnlyHint"),
            destructive=_attr(ann, "destructive_hint", "destructiveHint"),
            idempotent=_attr(ann, "idempotent_hint", "idempotentHint"),
            open_world=_attr(ann, "open_world_hint", "openWorldHint"),
        )

    @property
    def is_destructive(self) -> bool:
        """Per the MCP spec: destructive unless read-only or explicitly marked non-destructive."""
        return not self.read_only and self.destructive is not False

    @property
    def safe_to_repeat(self) -> bool:
        return bool(self.read_only or self.idempotent)


def render_content(blocks: Iterable[Any]) -> str:
    """Flatten MCP content blocks into text a model can read. Binary content becomes a short marker
    (it would only burn context); fetch it yourself with `read_resource` if you need the bytes."""
    parts: list[str] = []
    for b in blocks or []:
        kind = _attr(b, "type")
        if kind == "text":
            parts.append(str(_attr(b, "text", default="")))
        elif kind in ("image", "audio"):
            parts.append(f"[{kind}: {_attr(b, 'mime_type', 'mimeType', default='?')}, omitted]")
        elif kind == "resource":
            res = _attr(b, "resource")
            text = _attr(res, "text")
            parts.append(
                str(text)
                if text is not None
                else f"[resource {_attr(res, 'uri', default='?')}: binary, omitted]"
            )
        elif kind == "resource_link":
            parts.append(f"[resource link: {_attr(b, 'uri', default='?')}]")
        else:
            parts.append(f"[{kind or 'content'}]")
    return "\n".join(p for p in parts if p)


class MCPClient:
    """One connection to one MCP server. Use as an async context manager."""

    def __init__(
        self,
        session: Any | None = None,
        *,
        name: str = "mcp",
        _connect: Callable[[AsyncExitStack], Any] | None = None,
    ) -> None:
        self.name = name
        self._session, self._connect = session, _connect
        self._owned = session is None  # we only close connections we opened
        self._stack: AsyncExitStack | None = None
        self._tools: list[MCPToolInfo] | None = None

    # -- connecting ---------------------------------------------------------------------------
    @classmethod
    def stdio(
        cls,
        command: str,
        args: Sequence[str] = (),
        *,
        env: Mapping[str, str] | None = None,
        cwd: str | None = None,
        name: str | None = None,
    ) -> MCPClient:
        """Launch `command` as a subprocess and speak MCP over its stdin/stdout. The child gets only a
        small safe set of environment variables plus `env`: put secrets in `env` explicitly."""

        async def connect(stack: AsyncExitStack) -> Any:
            import mcp

            params = mcp.StdioServerParameters(
                command=command, args=list(args), env=dict(env) if env else None, cwd=cwd
            )
            if hasattr(mcp, "Client"):  # mcp >= 2
                return await stack.enter_async_context(mcp.Client(params))
            from mcp.client.stdio import stdio_client

            read, write = await stack.enter_async_context(stdio_client(params))
            return await _open_session(stack, read, write)

        return cls(name=name or _slug(command), _connect=connect)

    @classmethod
    def http(
        cls, url: str, *, headers: Mapping[str, str] | None = None, name: str | None = None
    ) -> MCPClient:
        """Connect to a Streamable HTTP MCP endpoint. `headers` carries credentials (e.g. Authorization)."""

        async def connect(stack: AsyncExitStack) -> Any:
            import mcp
            from mcp.client import streamable_http as transport

            if hasattr(mcp, "Client") and not headers:  # mcp >= 2: the simple form
                return await stack.enter_async_context(mcp.Client(url))
            if not hasattr(transport, "streamable_http_client"):  # early 1.x
                streams = await stack.enter_async_context(
                    transport.streamablehttp_client(url, headers=dict(headers) if headers else None)
                )
                return await _open_session(stack, streams[0], streams[1])
            http_client = None
            if headers:
                try:  # the 2.x SDK ships its own HTTP stack (httpx2); 1.x uses httpx
                    import httpx2 as hx
                except ImportError:
                    import httpx as hx
                http_client = hx.AsyncClient(
                    headers=dict(headers), timeout=hx.Timeout(30.0, read=300.0)
                )
                stack.push_async_callback(http_client.aclose)
            streams = await stack.enter_async_context(
                transport.streamable_http_client(
                    url, **({"http_client": http_client} if http_client else {})
                )
            )
            return await _open_session(stack, streams[0], streams[1])

        return cls(name=name or _slug(url), _connect=connect)

    async def __aenter__(self) -> Self:
        if self._session is None:
            assert self._connect is not None
            self._stack = AsyncExitStack()
            try:
                self._session = await self._connect(self._stack)
            except BaseException:
                await self._stack.aclose()
                self._stack = None
                raise
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._stack is not None:
            stack, self._stack = self._stack, None
            self._session = None
            await stack.aclose()

    @property
    def session(self) -> Any:
        if self._session is None:
            raise RuntimeError("MCPClient is not connected: use `async with`")
        return self._session

    # -- discovery ----------------------------------------------------------------------------
    async def _page(self, cursor: str | None) -> Any:
        params = inspect.signature(self.session.list_tools).parameters
        if "cursor" in params:
            return await self.session.list_tools(cursor=cursor)
        if cursor is None:
            return await self.session.list_tools()
        from mcp.types import PaginatedRequestParams

        return await self.session.list_tools(params=PaginatedRequestParams(cursor=cursor))

    async def list_tools(self, *, refresh: bool = False) -> list[MCPToolInfo]:
        """Every tool the server offers (all pages), cached until `refresh=True`."""
        if self._tools is None or refresh:
            out: list[MCPToolInfo] = []
            cursor: str | None = None
            while True:
                res = await self._page(cursor)
                out += [MCPToolInfo.of(t) for t in _attr(res, "tools", default=[])]
                cursor = _attr(res, "next_cursor", "nextCursor")
                if not cursor:
                    break
            self._tools = out
        return list(self._tools)

    async def call(self, name: str, arguments: dict[str, Any] | None = None) -> tuple[str, Any]:
        """Call a tool; returns `(text, structured)`. Raises `ToolError` if the server reports an error."""
        res = await self.session.call_tool(name, arguments or {})
        text = render_content(_attr(res, "content", default=[]))
        if _attr(res, "is_error", "isError", default=False):
            raise ToolError(text or f"MCP tool {name!r} reported an error")
        return text, _attr(res, "structured_content", "structuredContent")

    async def read_resource(self, uri: str) -> str:
        """The text of a resource (binary parts are summarized, not returned)."""
        res = await self.session.read_resource(uri)
        parts = []
        for c in _attr(res, "contents", default=[]):
            text = _attr(c, "text")
            parts.append(
                str(text)
                if text is not None
                else f"[binary resource {_attr(c, 'uri', default=uri)}, omitted]"
            )
        return "\n".join(parts)

    async def list_resources(self) -> list[dict[str, Any]]:
        res = await self.session.list_resources()
        return [
            {
                "uri": str(_attr(r, "uri")),
                "name": _attr(r, "name"),
                "description": _attr(r, "description"),
                "mime_type": _attr(r, "mime_type", "mimeType"),
            }
            for r in _attr(res, "resources", default=[])
        ]

    # -- tools --------------------------------------------------------------------------------
    async def load_tools(self, **kw: Any) -> list[Tool]:
        """`list_tools()` then `tools(...)`: the usual way to fill a registry."""
        await self.list_tools()
        return self.tools(**kw)

    def tools(
        self,
        *,
        prefix: str | None = None,
        include: Iterable[str] | None = None,
        exclude: Iterable[str] = (),
        approval: Approval = "none",
        trust_annotations: bool = True,
        timeout_s: float = 60.0,
        max_result_chars: int | None = 20_000,
        infos: Sequence[MCPToolInfo] | None = None,
    ) -> list[Tool]:
        """The server's tools as AAA tools (call `list_tools()` first, or use `load_tools`).

        `include`/`exclude` are glob patterns on the server's tool names. `approval="destructive"` requires
        human approval for every tool that is not read-only (MCP: destructive unless stated otherwise);
        `"always"` for all. `trust_annotations=False` ignores the server's hints and treats every tool as
        non-repeatable (`at_most_once`) and destructive."""
        listed = list(infos) if infos is not None else self._tools
        if listed is None:
            raise RuntimeError("call `await client.list_tools()` (or `load_tools`) first")
        pre = f"{self.name}__" if prefix is None else prefix
        inc, exc = list(include) if include is not None else None, list(exclude)
        out: list[Tool] = []
        for info in listed:
            if inc is not None and not any(fnmatch.fnmatchcase(info.name, p) for p in inc):
                continue
            if any(fnmatch.fnmatchcase(info.name, p) for p in exc):
                continue
            hint = info if trust_annotations else MCPToolInfo(name=info.name)
            needs_approval = approval == "always" or (
                approval == "destructive" and hint.is_destructive
            )
            out.append(
                SchemaTool(
                    self._caller(info.name, max_result_chars),
                    name=mcp_tool_name(pre, info.name),
                    description=(info.description or info.title or info.name).strip(),
                    parameters=info.parameters,
                    requires_approval=needs_approval,
                    timeout_s=timeout_s,
                    pool=f"mcp:{self.name}",
                    delivery="at_least_once" if hint.safe_to_repeat else "at_most_once",
                )
            )
        return out

    def _caller(self, remote_name: str, limit: int | None) -> Callable[[dict[str, Any]], Any]:
        async def call(arguments: dict[str, Any]) -> Any:
            text, structured = await self.call(remote_name, arguments)
            result: Any = text if text else structured
            if isinstance(result, str) and limit is not None and len(result) > limit:
                result = f"{result[:limit]}\n[... truncated {len(result) - limit} characters]"
            return result

        return call


async def _open_session(stack: AsyncExitStack, read: Any, write: Any) -> Any:
    import mcp

    session = await stack.enter_async_context(mcp.ClientSession(read, write))
    await session.initialize()
    return session


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")[:24] or "mcp"
