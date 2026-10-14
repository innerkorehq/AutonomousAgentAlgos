"""Vocabulary shared by every memory backend.

Memory libraries disagree about almost everything (Mem0: `user_id`/`agent_id`/`run_id`; Graphiti:
`group_id`; Supermemory: `containerTag`; LangGraph stores: a namespace tuple). `Scope` is the one
place AAA states *whose* memory something is, and each adapter maps it onto its native notion.

A scope is always built by the application (from the run's tenant and metadata), never from a value
the model produced: that is what keeps one customer's memories out of another's prompts."""

from __future__ import annotations

import hashlib
import re
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

MemoryKind = Literal["fact", "episode", "profile", "procedure", "document"]


def encode_part(value: str) -> str:
    """Reversible, injective encoding to `[A-Za-z0-9_]`: every other character becomes `_<hex>_`.
    (So `-` never appears inside an encoded part and can safely delimit parts.)"""
    return re.sub(r"[^A-Za-z0-9]", lambda m: f"_{ord(m.group()):x}_", value)


class Scope(BaseModel):
    """Whose memory. All fields optional; the more that are set, the narrower the scope.

    `tenant` is the isolation boundary; `user` and `agent` partition inside it; `session` is a
    single conversation or run; `namespace` is free-form extra partitioning (e.g. a project)."""

    model_config = ConfigDict(frozen=True)

    tenant: str | None = None
    user: str | None = None
    agent: str | None = None
    session: str | None = None
    namespace: tuple[str, ...] = ()

    def key(self, *, include_session: bool = False) -> str:
        """A stable string for this scope, safe to use as a group id / container tag / key prefix.
        The session is excluded by default: long-term memory outlives the conversation that made it."""
        parts = [
            f"t:{self.tenant}" if self.tenant else "",
            f"u:{self.user}" if self.user else "",
            f"a:{self.agent}" if self.agent else "",
            f"s:{self.session}" if include_session and self.session else "",
            *[f"n:{n}" for n in self.namespace],
        ]
        return "|".join(p for p in parts if p) or "global"

    def slug(self, *, max_len: int | None = None) -> str:
        """An identifier made only of `[A-Za-z0-9_-]`, unique per (tenant, user, agent, namespace),
        for backends that restrict ids (Graphiti group ids, Supermemory container tags, LangGraph
        namespace labels, Mem0 entity ids). Parts are ordered tenant, user, agent, namespace, so a
        tenant's slugs all start with `tenant_prefix()`. Long slugs are shortened with a hash tail."""
        parts = [
            f"t-{encode_part(self.tenant)}" if self.tenant else "",
            f"u-{encode_part(self.user)}" if self.user else "",
            f"a-{encode_part(self.agent)}" if self.agent else "",
            *[f"n-{encode_part(n)}" for n in self.namespace],
        ]
        out = "--".join(p for p in parts if p) or "global"
        if max_len is not None and len(out) > max_len:
            tail = hashlib.sha256(out.encode()).hexdigest()[:16]
            out = f"{out[: max_len - 17]}-{tail}"
        return out

    def tenant_prefix(self) -> str:
        """The slug prefix shared by everything in this scope's tenant."""
        return f"t-{encode_part(self.tenant)}" if self.tenant else "global"

    def contains(self, other: Scope) -> bool:
        """True if `other` is this scope or narrower (so forgetting `self` removes `other`'s data).
        The tenant must match exactly (an untenanted scope never reaches into a tenant's data)."""
        if self.tenant != other.tenant:
            return False
        for f in ("user", "agent"):
            mine = getattr(self, f)
            if mine is not None and mine != getattr(other, f):
                return False
        return other.namespace[: len(self.namespace)] == self.namespace


class MemoryEntry(BaseModel):
    """One thing to remember: a statement, or a conversation turn (`role` set)."""

    text: str
    role: str | None = None  # "user" | "assistant" | ... when this is a conversation turn
    kind: MemoryKind = "fact"
    id: str | None = None  # a stable id enables idempotent writes on backends that support one
    time: datetime | None = None  # when it happened (episodes); defaults to now
    meta: dict[str, Any] = Field(default_factory=dict)


class MemoryItem(BaseModel):
    """One recalled memory."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    text: str
    kind: MemoryKind = "fact"
    score: float | None = None
    created_at: datetime | None = None
    valid_at: datetime | None = None  # temporal backends (Graphiti): when the fact became true...
    invalid_at: datetime | None = None  # ...and when it stopped being true (None = still true)
    meta: dict[str, Any] = Field(default_factory=dict)
    raw: Any = Field(default=None, exclude=True)  # the backend's own object, for escape hatches

    def valid_on(self, when: datetime) -> bool:
        if self.valid_at is not None and when < self.valid_at:
            return False
        return not (self.invalid_at is not None and when >= self.invalid_at)


class AddResult(BaseModel):
    ids: list[str] = Field(default_factory=list)
    pending: bool = False  # the backend ingests asynchronously: stored, not yet searchable
    usage: dict[str, int] = Field(default_factory=dict)  # tokens the backend spent, if it says
    cost: float = 0.0
    detail: dict[str, Any] = Field(default_factory=dict)


def render_memories(items: list[MemoryItem], *, max_chars: int = 2000) -> str:
    """Compact bullet list for a prompt; stops before exceeding `max_chars`."""
    lines: list[str] = []
    used = 0
    for m in items:
        when = f" (since {m.valid_at:%Y-%m-%d})" if m.valid_at else ""
        line = f"- {m.text}{when}"
        if used + len(line) > max_chars and lines:
            break
        lines.append(line)
        used += len(line) + 1
    return "\n".join(lines)
