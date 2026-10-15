"""Attachments on a message: images, audio and files.

A `Message` keeps its text in `content`; `parts` carries attachments that come AFTER that text, in order:

    Message(role="user", content="What is in this picture?", parts=[ImagePart.from_path("cat.png")])

Where the bytes live:

* in memory, as base64 `data`, while you build the message;
* in the engine's blob store as a `blob` reference once an event holding them is appended (the log never keeps large base64),
  see `Engine(blobs=...)`;
* as a `url` the provider will fetch itself (only https/http, see docs/security/multimodal.md).

The checks here (size, declared type vs the kind, image signature, URL scheme) happen when a part is created or validated, so a bad attachment fails
at the edge, not inside a provider call."""

from __future__ import annotations

import base64
import binascii
import mimetypes
import posixpath
from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, Field, field_validator, model_validator

from aaa.events.event import BlobRef

MAX_MEDIA_BYTES = 20 * 1024 * 1024
MAX_INLINE_NO_BLOBS = (
    256 * 1024
)  # without a blob store, attachments above this are refused (the log would carry them)
URL_SCHEMES = ("https",)

# leading bytes of common image formats; checked when the declared type is one of these and the bytes are at hand
_SIGNATURES: dict[str, tuple[tuple[int, bytes], ...]] = {
    "image/png": ((0, b"\x89PNG\r\n\x1a\n"),),
    "image/jpeg": ((0, b"\xff\xd8\xff"),),
    "image/gif": ((0, b"GIF87a"), (0, b"GIF89a")),
    "image/webp": ((8, b"WEBP"),),
}


class MediaError(ValueError):
    """An attachment was refused (size, type, signature, URL scheme)."""


class MediaPart(BaseModel):
    kind: ClassVar[str] = ""
    type: str
    media_type: str
    data: str | None = None  # base64
    url: str | None = None
    blob: BlobRef | None = None
    size: int | None = None  # decoded size in bytes, set when known
    name: str | None = None  # a file name for display only; never used as a path

    @field_validator("name")
    @classmethod
    def _base_name(cls, v: str | None) -> str | None:
        return (
            posixpath.basename(v.replace("\\", "/")) if v else v
        )  # metadata only: strip any directory part

    @model_validator(mode="after")
    def _check(self) -> MediaPart:
        if sum(x is not None for x in (self.data, self.url, self.blob)) != 1:
            raise MediaError("an attachment needs exactly one of data, url or blob")
        if (cls_kind := self.kind) and not self.media_type.startswith(f"{cls_kind}/"):
            raise MediaError(f"a {cls_kind} attachment cannot have media type {self.media_type!r}")
        if self.url is not None:
            scheme = urlsplit(self.url).scheme.lower()
            if scheme not in URL_SCHEMES:
                raise MediaError(
                    f"attachment URL scheme {scheme!r} is not allowed (allowed: {', '.join(URL_SCHEMES)})"
                )
        if self.data is not None:
            try:
                raw = base64.b64decode(self.data, validate=True)
            except (binascii.Error, ValueError) as e:
                raise MediaError("attachment data is not valid base64") from e
            if len(raw) > MAX_MEDIA_BYTES:
                raise MediaError(f"attachment is {len(raw)} bytes; the limit is {MAX_MEDIA_BYTES}")
            sniff(self.media_type, raw)
            self.size = len(raw)
        if self.blob is not None and self.blob.size > MAX_MEDIA_BYTES:
            raise MediaError(
                f"attachment is {self.blob.size} bytes; the limit is {MAX_MEDIA_BYTES}"
            )
        return self

    # -- construction ---------------------------------------------------------------------------
    @classmethod
    def from_bytes(cls, raw: bytes, media_type: str, **kw: Any) -> Any:
        return cls(media_type=media_type, data=base64.b64encode(raw).decode(), **kw)

    @classmethod
    def from_path(cls, path: str | Path, media_type: str | None = None, **kw: Any) -> Any:
        p = Path(path)
        mt = media_type or mimetypes.guess_type(p.name)[0] or "application/octet-stream"
        if p.stat().st_size > MAX_MEDIA_BYTES:
            raise MediaError(
                f"{p.name} is {p.stat().st_size} bytes; the limit is {MAX_MEDIA_BYTES}"
            )
        kw.setdefault("name", p.name)
        return cls.from_bytes(p.read_bytes(), mt, **kw)

    @classmethod
    def from_url(cls, url: str, media_type: str, **kw: Any) -> Any:
        return cls(media_type=media_type, url=url, **kw)


def sniff(media_type: str, raw: bytes) -> None:
    """Refuse bytes whose leading signature contradicts a declared common image type. We never decode media."""
    for offset, sig in _SIGNATURES.get(media_type, ()):
        if raw[offset : offset + len(sig)] == sig:
            return
    if media_type in _SIGNATURES:
        raise MediaError(f"the bytes are not a valid {media_type} (signature mismatch)")


class ImagePart(MediaPart):
    kind: ClassVar[str] = "image"
    type: Literal["image"] = "image"
    detail: Literal["auto", "low", "high"] | None = None


class AudioPart(MediaPart):
    kind: ClassVar[str] = "audio"
    type: Literal["audio"] = "audio"


class FilePart(MediaPart):
    kind: ClassVar[str] = ""
    type: Literal["file"] = "file"


Part = Annotated[ImagePart | AudioPart | FilePart, Field(discriminator="type")]
MEDIA_TYPES = ("image", "audio", "file")


def is_media_dict(d: Any) -> bool:
    """A JSON object that looks like an attachment (used by the engine and the redactor, which see plain dicts)."""
    return isinstance(d, dict) and d.get("type") in MEDIA_TYPES and "media_type" in d


def kinds(parts: list[Any] | None) -> set[str]:
    return {p.type for p in parts or []}


def describe(p: MediaPart) -> str:
    """A one-line stand-in for an attachment in text-only contexts (summaries, traces, token-saving placeholders)."""
    where = "url" if p.url else "blob" if p.blob else "inline"
    size = p.size if p.size is not None else (p.blob.size if p.blob else None)
    return (
        f"[{p.type} attachment: {p.media_type}"
        + (f", {size} bytes" if size is not None else "")
        + f", {where}]"
    )


# -- stream-salted blobs ---------------------------------------------------------------------------------
_MAGIC = b"AAAM1\x00"


def salt(stream: str, raw: bytes) -> bytes:
    """Prefix the bytes with their stream id, so identical media in two runs are two different content-addressed blobs
    (erasing one run can never delete, or reveal the existence of, another's)."""
    return _MAGIC + stream.encode() + b"\x00" + raw


def unsalt(blob: bytes) -> bytes:
    if not blob.startswith(_MAGIC):
        raise MediaError("not an attachment blob")
    return blob.split(b"\x00", 2)[2]
