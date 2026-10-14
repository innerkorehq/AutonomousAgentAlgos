from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel


class StoredSnapshot(BaseModel):
    """A projection's state as of `seq` (inclusive). Pure derived data: always safe to drop."""

    stream: str
    name: str
    seq: int
    version: int
    state: str  # JSON


@runtime_checkable
class SnapshotCapable(Protocol):
    async def load_snapshot(self, stream: str, name: str) -> StoredSnapshot | None: ...

    async def save_snapshot(self, snap: StoredSnapshot) -> None:
        """Must be monotonic: ignore a snapshot whose seq is not newer than the stored one."""
        ...
