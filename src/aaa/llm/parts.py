"""Attachments on a message (images, audio, files). The implementation lives in `aaa.core.parts` so the runtime and the redactor can use it
without importing the LLM package; this is its public home."""

from aaa.core.parts import (
    MAX_INLINE_NO_BLOBS,
    MAX_MEDIA_BYTES,
    MEDIA_TYPES,
    URL_SCHEMES,
    AudioPart,
    FilePart,
    ImagePart,
    MediaError,
    MediaPart,
    Part,
    describe,
    is_media_dict,
    kinds,
    salt,
    sniff,
    unsalt,
)

__all__ = [
    "MAX_INLINE_NO_BLOBS",
    "MAX_MEDIA_BYTES",
    "MEDIA_TYPES",
    "URL_SCHEMES",
    "AudioPart",
    "FilePart",
    "ImagePart",
    "MediaError",
    "MediaPart",
    "Part",
    "describe",
    "is_media_dict",
    "kinds",
    "salt",
    "sniff",
    "unsalt",
]
