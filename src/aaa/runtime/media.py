"""Move attachment bytes out of events, before they are appended.

Called by `Engine.emit` ahead of the redactor and the claim-check. Any attachment dict (see `aaa.llm.parts`) found in an event's data is validated
(size, declared type, image signature, URL scheme); bytes at or above `threshold` go to the blob store, salted with the stream id, and the part is
rewritten to carry a `blob` reference instead of base64. The references are recorded in `event.blobs`, which is what `engine.erase` deletes."""

from __future__ import annotations

import base64
import json
from typing import Any

from pydantic import TypeAdapter

from aaa.core import parts as P
from aaa.data.blob import BlobStore
from aaa.events.event import BlobRef, Event

_PART = TypeAdapter(P.Part)
THRESHOLD = 4096  # decoded bytes: smaller attachments stay inline in the event


def _looks_like_media(raw: str) -> bool:
    return '"data"' in raw and any(f'"type": "{k}"' in raw for k in P.MEDIA_TYPES)


async def offload(event: Event, blobs: BlobStore | None, threshold: int = THRESHOLD) -> Event:
    raw = json.dumps(event.data, default=str)
    if not _looks_like_media(raw):
        return event
    refs: list[BlobRef] = []

    async def walk(v: Any) -> Any:
        if P.is_media_dict(v) and v.get("data") is not None:
            part = _PART.validate_python(v)  # raises on a bad attachment, at the edge
            size = part.size or 0
            if blobs is None:
                if size > P.MAX_INLINE_NO_BLOBS:
                    raise P.MediaError(
                        f"attachment is {size} bytes and this engine has no blob store; attachments over {P.MAX_INLINE_NO_BLOBS} bytes need Engine(blobs=FileBlobStore(...))"
                    )
                return v
            if size < threshold:
                return v
            ref = await blobs.put(
                P.salt(event.stream, base64.b64decode(v["data"])), v["media_type"]
            )
            refs.append(ref)
            out = {k: x for k, x in v.items() if k != "data"}
            out["blob"] = ref.model_dump(mode="json")
            out["size"] = size
            return out
        if isinstance(v, dict):
            return {k: await walk(x) for k, x in v.items()}
        if isinstance(v, list):
            return [await walk(x) for x in v]
        return v

    data = await walk(event.data)
    if not refs:
        return event
    return event.model_copy(update={"data": data, "blobs": (*event.blobs, *refs)})


async def read(part: P.MediaPart, blobs: BlobStore | None) -> bytes:
    """The attachment's bytes (not available for a URL attachment: the provider fetches those)."""
    if part.data is not None:
        return base64.b64decode(part.data)
    if part.blob is not None:
        if blobs is None:
            raise P.MediaError("the attachment is in a blob store but this engine has none")
        return P.unsalt(await blobs.get(part.blob))
    raise P.MediaError("a URL attachment has no bytes here: the provider fetches it")
