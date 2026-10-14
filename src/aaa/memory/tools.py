"""Tools that let the model manage its own memory (the "hot path" style, as in LangMem and Letta):

    registry = ToolRegistry([*memory_tools(memory), search_the_web])

The model chooses WHAT to remember and WHEN to recall; it never chooses WHOSE memory. The scope is
built from the run's tenant and metadata (`ToolContext.tenant`, `ToolContext.meta`), so a prompt
injection cannot read or write another customer's memories."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from aaa.memory.base import Deletable, Memory, is_idempotent
from aaa.memory.types import MemoryEntry, Scope
from aaa.tools.tool import Tool, ToolContext, tool


def scope_from_tool_context(ctx: ToolContext) -> Scope:
    user, agent = ctx.meta.get("user"), ctx.meta.get("agent")
    return Scope(
        tenant=ctx.tenant,
        user=str(user) if user is not None else None,
        agent=str(agent) if agent is not None else None,
        session=ctx.run_id,
    )


def memory_tools(
    memory: Memory,
    *,
    k: int = 5,
    scope_for: Callable[[ToolContext], Scope] = scope_from_tool_context,
    allow_forget: bool = False,
) -> list[Tool]:
    """`remember(text)`, `recall(query)` and (opt-in) `forget(memory_id)` tools bound to `memory`."""
    # A backend that cannot de-duplicate must not have its writes re-run after a worker dies.
    delivery = "at_least_once" if is_idempotent(memory) else "at_most_once"

    @tool(delivery=delivery)
    async def remember(text: str, ctx: ToolContext) -> dict[str, Any]:
        """Save a durable fact worth knowing in future conversations (a preference, a decision, a
        constraint). Write one short, self-contained statement."""
        res = await memory.add([MemoryEntry(text=text)], scope_for(ctx), key=ctx.idempotency_key)
        return {"saved": bool(res.ids) or res.pending, "ids": res.ids}

    @tool
    async def recall(query: str, ctx: ToolContext) -> list[dict[str, Any]]:
        """Search long-term memory for facts relevant to the query. Use it before answering
        anything that may depend on what is already known about the user or earlier work."""
        items = await memory.search(query, scope_for(ctx), k=k)
        return [
            {
                "id": m.id,
                "memory": m.text,
                **({"since": m.valid_at.date().isoformat()} if m.valid_at else {}),
                **({"until": m.invalid_at.date().isoformat()} if m.invalid_at else {}),
            }
            for m in items
        ]

    tools: list[Tool] = [remember, recall]
    if allow_forget:
        if not isinstance(memory, Deletable):
            raise TypeError(
                f"{type(memory).__name__} cannot delete, so allow_forget is unavailable"
            )
        deleter: Deletable = memory

        @tool(name="forget", delivery="at_least_once")
        async def forget(memory_id: str, ctx: ToolContext) -> dict[str, int]:
            """Delete one saved memory by its id (as returned by recall)."""
            return {"deleted": await deleter.delete([memory_id], scope_for(ctx))}

        tools.append(forget)
    return tools
