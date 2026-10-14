from __future__ import annotations

import asyncio
import hashlib
import os
import tempfile
from pathlib import Path
from typing import Protocol, runtime_checkable

from aaa.events.event import BlobRef


class BlobIntegrityError(Exception):
    pass


@runtime_checkable
class BlobStore(Protocol):
    """Content-addressed (sha256) storage for payloads too large to live in an event."""

    async def put(self, data: bytes, media_type: str = "application/octet-stream") -> BlobRef: ...
    async def get(self, ref: BlobRef) -> bytes: ...
    async def exists(self, ref: BlobRef) -> bool: ...
    async def delete(self, ref: BlobRef) -> None: ...


def _verify(ref: BlobRef, data: bytes) -> bytes:
    if hashlib.sha256(data).hexdigest() != ref.sha256:
        raise BlobIntegrityError(f"blob {ref.sha256[:12]} failed integrity check")
    return data


class InMemoryBlobStore:
    def __init__(self) -> None:
        self._d: dict[str, bytes] = {}

    async def put(self, data: bytes, media_type: str = "application/octet-stream") -> BlobRef:
        h = hashlib.sha256(data).hexdigest()
        self._d[h] = data  # idempotent: same content -> same key (dedup)
        return BlobRef(uri=f"mem://sha256/{h}", sha256=h, size=len(data), media_type=media_type)

    async def get(self, ref: BlobRef) -> bytes:
        try:
            return _verify(ref, self._d[ref.sha256])
        except KeyError:
            raise FileNotFoundError(ref.uri) from None

    async def exists(self, ref: BlobRef) -> bool:
        return ref.sha256 in self._d

    async def delete(self, ref: BlobRef) -> None:
        self._d.pop(ref.sha256, None)


class FileBlobStore:
    """Local-disk store, sharded by hash prefix, atomic writes, verified reads."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def _path(self, sha: str) -> Path:
        return self.root / sha[:2] / sha[2:4] / sha

    async def put(self, data: bytes, media_type: str = "application/octet-stream") -> BlobRef:
        h = hashlib.sha256(data).hexdigest()
        p = self._path(h)

        def write() -> None:
            if p.exists():
                return
            p.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=p.parent)
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            os.replace(tmp, p)

        await asyncio.to_thread(write)
        return BlobRef(uri=p.as_uri(), sha256=h, size=len(data), media_type=media_type)

    async def get(self, ref: BlobRef) -> bytes:
        return _verify(ref, await asyncio.to_thread(self._path(ref.sha256).read_bytes))

    async def exists(self, ref: BlobRef) -> bool:
        return await asyncio.to_thread(self._path(ref.sha256).exists)

    async def delete(self, ref: BlobRef) -> None:
        await asyncio.to_thread(self._path(ref.sha256).unlink, missing_ok=True)
