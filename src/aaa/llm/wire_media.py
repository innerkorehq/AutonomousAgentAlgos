"""How attachments are written on the wire for OpenAI-style APIs (chat completions and the Responses API).

By the time a client sees a message, the handler has resolved blob references into inline base64 `data` (or the part carries a provider-fetched `url`)."""

from __future__ import annotations

from typing import Any

from aaa.llm.errors import FatalError
from aaa.llm.parts import MediaPart
from aaa.llm.types import Message

_AUDIO_FORMATS = {
    "audio/wav": "wav",
    "audio/x-wav": "wav",
    "audio/wave": "wav",
    "audio/mpeg": "mp3",
    "audio/mp3": "mp3",
}


def _data_url(p: MediaPart) -> str:
    return p.url or f"data:{p.media_type};base64,{p.data}"


def _need_bytes(p: MediaPart) -> str:
    if p.data is None:
        raise FatalError(f"this API needs the bytes of a {p.type} attachment, not a URL")
    return p.data


def chat_content(m: Message) -> Any:
    """`content` for a chat-completions message: the plain string, or a list of typed parts when it carries attachments."""
    if not m.parts:
        return m.content
    items: list[dict[str, Any]] = [{"type": "text", "text": m.content}] if m.content else []
    for p in m.parts:
        if p.type == "image":
            img: dict[str, Any] = {"url": _data_url(p)}
            if getattr(p, "detail", None):
                img["detail"] = p.detail
            items.append({"type": "image_url", "image_url": img})
        elif p.type == "audio":
            fmt = _AUDIO_FORMATS.get(p.media_type)
            if fmt is None:
                raise FatalError(f"chat completions accepts wav and mp3 audio, not {p.media_type}")
            items.append(
                {"type": "input_audio", "input_audio": {"data": _need_bytes(p), "format": fmt}}
            )
        else:
            file: dict[str, Any] = {"file_data": f"data:{p.media_type};base64,{_need_bytes(p)}"}
            if p.name:
                file["filename"] = p.name
            items.append({"type": "file", "file": file})
    return items


def responses_content(m: Message) -> Any:
    """`content` for a Responses-API user message."""
    if not m.parts:
        return m.content or ""
    items: list[dict[str, Any]] = [{"type": "input_text", "text": m.content}] if m.content else []
    for p in m.parts:
        if p.type == "image":
            img: dict[str, Any] = {"type": "input_image", "image_url": _data_url(p)}
            if getattr(p, "detail", None):
                img["detail"] = p.detail
            items.append(img)
        elif p.type == "file":
            f: dict[str, Any] = {"type": "input_file"}
            if p.url:
                f["file_url"] = p.url
            else:
                f["file_data"] = f"data:{p.media_type};base64,{_need_bytes(p)}"
            if p.name:
                f["filename"] = p.name
            items.append(f)
        else:
            raise FatalError(f"the Responses API client does not send {p.type} attachments")
    return items
