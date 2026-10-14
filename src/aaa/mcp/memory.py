"""Use any MCP memory server as an AAA `Memory`.

Memory servers expose their operations as tools, and the tool names and argument shapes differ from
server to server (`add_memory` / `add_memories` / `add_episode`, `search_memory` / `search_nodes` ...).
There is deliberately no built-in table of servers: you say which tool does what, and (only if the
defaults do not match) how to build its arguments and read its answer. List a server's real tools
with `await client.list_tools()` first.

    memory = mcp_memory(
        client,
        add="add_memory",
        search="search_memory",
        forget="delete_all_memories",          # optional: switches on erasure support
    )

Defaults send `{"content", "user_id"}` to add, `{"query", "user_id", "limit"}` to search and
`{"user_id"}` to forget, where `user_id` is the scope's slug, and read results that are JSON lists,
JSON objects holding a list (`results`/`memories`/`facts`/`items`/`nodes`), or plain text lines.

Isolation caveat: an MCP server is a separate process with its own idea of tenancy. AAA passes the scope as
an argument, but it is the SERVER that must honour it. For hard tenant isolation, connect one server
(or one credential) per tenant instead of sharing one."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import Any

from aaa.mcp.client import MCPClient
from aaa.memory.callable import CallableMemory, coerce_item
from aaa.memory.types import AddResult, MemoryEntry, MemoryItem, Scope

_LIST_KEYS = ("results", "memories", "facts", "items", "nodes", "entries", "data")

AddArgs = Callable[[Sequence[MemoryEntry], Scope], dict[str, Any]]
SearchArgs = Callable[[str, Scope, int], dict[str, Any]]
ForgetArgs = Callable[[Scope], dict[str, Any]]
DeleteArgs = Callable[[Sequence[str], Scope], dict[str, Any]]
Parse = Callable[[str, Any], list[MemoryItem]]


def default_add_args(entries: Sequence[MemoryEntry], scope: Scope) -> dict[str, Any]:
    return {"content": "\n".join(e.text for e in entries), "user_id": scope.slug()}


def default_search_args(query: str, scope: Scope, k: int) -> dict[str, Any]:
    return {"query": query, "user_id": scope.slug(), "limit": k}


def default_forget_args(scope: Scope) -> dict[str, Any]:
    return {"user_id": scope.slug()}


def default_delete_args(ids: Sequence[str], scope: Scope) -> dict[str, Any]:
    return {"ids": list(ids), "user_id": scope.slug()}


def _unwrap(data: Any) -> Any:
    """Servers return a list, an object holding one, or wrap a bare value as `{"result": value}`
    (FastMCP does this for plain return types); JSON that arrives as a string is decoded."""
    for _ in range(3):
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except ValueError:
                return data
        elif isinstance(data, dict):
            inner = next((data[k] for k in ("result", *_LIST_KEYS) if k in data), None)
            if inner is None:
                return data
            data = inner
        else:
            break
    return data


def default_parse(text: str, structured: Any) -> list[MemoryItem]:
    """Best-effort reading of a search answer: structured content, JSON text, or lines of text."""
    for source in (structured, text):
        data = _unwrap(source) if source is not None else None
        if isinstance(data, list):
            return [coerce_item(x) for x in data]
    return [coerce_item(line.lstrip("-* ").strip()) for line in text.splitlines() if line.strip()]


def mcp_memory(
    client: MCPClient,
    *,
    add: str,
    search: str,
    forget: str | None = None,
    delete: str | None = None,
    build_add: AddArgs = default_add_args,
    build_search: SearchArgs = default_search_args,
    build_forget: ForgetArgs = default_forget_args,
    build_delete: DeleteArgs = default_delete_args,
    parse: Parse = default_parse,
    idempotent: bool = False,
) -> CallableMemory:
    """A `Memory` backed by the named tools of `client`. See the module docstring."""

    async def do_add(entries: Sequence[MemoryEntry], scope: Scope) -> AddResult:
        await client.call(add, build_add(entries, scope))
        return AddResult()

    async def do_search(query: str, scope: Scope, k: int) -> list[MemoryItem]:
        text, structured = await client.call(search, build_search(query, scope, k))
        return parse(text, structured)[:k]

    async def do_forget(scope: Scope) -> int:
        await client.call(forget or "", build_forget(scope))
        return 0  # MCP servers rarely report how many they removed

    async def do_delete(ids: Sequence[str], scope: Scope) -> int:
        await client.call(delete or "", build_delete(ids, scope))
        return len(ids)

    return CallableMemory(
        add=do_add,
        search=do_search,
        forget=do_forget if forget else None,
        delete=do_delete if delete else None,
        idempotent=idempotent,
    )
