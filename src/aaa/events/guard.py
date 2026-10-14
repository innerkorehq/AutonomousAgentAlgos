"""Execution guards: mutual exclusion for side-effecting work across workers.

A guard answers "may I run THIS call right now?". It is a lease with a heartbeat: acquire it,
renew it while working, and treat a failed renewal as "I am no longer the only one running this".
`LocalGuard` covers a single process; `LeaseGuard` reuses the bus backend's partition-lease
primitive (Postgres advisory lock / Redis SET NX PX / memory) so it works across processes with
exactly the same failure semantics as partition ownership."""

from __future__ import annotations

import zlib
from typing import Protocol, runtime_checkable


@runtime_checkable
class ExecutionGuard(Protocol):
    async def acquire(self, key: str, ttl_s: float) -> bool: ...

    async def renew(self, key: str, ttl_s: float) -> bool:
        """True while this holder still owns `key`; False once ownership was lost."""
        ...

    async def release(self, key: str) -> None: ...


class LocalGuard:
    """Single-process exclusion. (Across a restart nothing is held, which is correct: the
    previous process is gone.)"""

    def __init__(self) -> None:
        self._held: set[str] = set()

    async def acquire(self, key: str, ttl_s: float) -> bool:
        if key in self._held:
            return False
        self._held.add(key)
        return True

    async def renew(self, key: str, ttl_s: float) -> bool:
        return key in self._held

    async def release(self, key: str) -> None:
        self._held.discard(key)


class LeaseGuard:
    """Cross-process exclusion via the bus backend's lease primitive."""

    def __init__(self, backend: object, member: str) -> None:
        self._b, self._member = backend, member

    @staticmethod
    def _addr(key: str) -> tuple[str, int]:
        # A 62-bit address (group string + partition number) so unrelated calls practically never
        # share a lease; the backends turn these into an advisory-lock / key / dict slot.
        return f"aaa.exec.{key}", zlib.crc32(key.encode()[::-1]) & 0x7FFFFFFF

    async def acquire(self, key: str, ttl_s: float) -> bool:
        g, p = self._addr(key)
        return bool(await self._b.try_lease(g, p, self._member, ttl_s))  # type: ignore[attr-defined]

    async def renew(self, key: str, ttl_s: float) -> bool:
        return await self.acquire(key, ttl_s)  # try_lease renews when we already hold it

    async def release(self, key: str) -> None:
        g, p = self._addr(key)
        await self._b.release_lease(g, p, self._member)  # type: ignore[attr-defined]
