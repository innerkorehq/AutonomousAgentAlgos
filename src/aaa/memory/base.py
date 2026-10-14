"""The memory protocol: a tiny required core plus optional capabilities.

Backends differ a lot (Mem0 extracts facts with its own LLM, Graphiti builds a temporal graph,
LangGraph stores are plain namespaced key/value with search), so the required surface is only
`add` and `search`. Everything else is a capability a backend may or may not have, discovered with
`isinstance` and checked up front with `require`, the same pattern as `aaa.data.capabilities`."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from aaa.core.errors import AAAError
from aaa.memory.types import AddResult, MemoryEntry, MemoryItem, Scope


class MissingMemoryCapability(AAAError):
    """A feature needs a capability the configured memory backend does not provide."""


@runtime_checkable
class Memory(Protocol):
    """Long-term memory. Implementations must be safe to call concurrently."""

    async def add(
        self, entries: Sequence[MemoryEntry], scope: Scope, *, key: str | None = None
    ) -> AddResult:
        """Remember `entries` for `scope`. `key` is an idempotency key: repeating a call with the
        same key must not store the same thing twice (backends that cannot honour that set
        `idempotent = False`, and AAA then refuses to retry their writes automatically)."""
        ...

    async def search(
        self,
        query: str,
        scope: Scope,
        *,
        k: int = 5,
        filters: dict[str, Any] | None = None,
        as_of: datetime | None = None,
    ) -> list[MemoryItem]:
        """The `k` memories of `scope` most relevant to `query`, best first. `as_of` asks for what
        was true at that moment (only meaningful on temporal backends; others ignore it)."""
        ...

    async def aclose(self) -> None: ...


@runtime_checkable
class Updatable(Protocol):
    async def update(self, memory_id: str, text: str, scope: Scope) -> MemoryItem: ...


@runtime_checkable
class Deletable(Protocol):
    async def delete(self, ids: Sequence[str], scope: Scope) -> int:
        """Delete memories by id; returns how many were removed."""
        ...


@runtime_checkable
class Forgettable(Protocol):
    async def forget(self, scope: Scope) -> int:
        """Delete everything stored for `scope` and anything narrower (right to erasure)."""
        ...


@runtime_checkable
class Profiled(Protocol):
    async def profile(self, scope: Scope) -> dict[str, Any]:
        """A synthesized summary of the scope (stable facts, recent context...)."""
        ...


@runtime_checkable
class Listable(Protocol):
    async def list(self, scope: Scope, *, limit: int = 100) -> list[MemoryItem]: ...


def supports(memory: object, capability: type) -> bool:
    return isinstance(memory, capability)


def require(memory: object, *capabilities: type) -> None:
    """Raise `MissingMemoryCapability` naming everything `memory` lacks (fail at startup, not mid-run)."""
    missing = [c.__name__ for c in capabilities if not isinstance(memory, c)]
    if missing:
        raise MissingMemoryCapability(
            f"{type(memory).__name__} does not provide: {', '.join(missing)}"
        )


def is_idempotent(memory: object) -> bool:
    """Whether repeating a keyed `add` is safe. Defaults to False: assume the worst."""
    return bool(getattr(memory, "idempotent", False))
