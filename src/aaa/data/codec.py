from __future__ import annotations

import json
from typing import Any, Generic, Protocol, TypeVar, cast, runtime_checkable

from pydantic import TypeAdapter

from aaa.data.item import Item, fingerprint_json

T = TypeVar("T")


@runtime_checkable
class Codec(Protocol[T]):
    """Teaches AAA how to render, persist and describe a user type. One per `kind`."""

    def to_text(self, value: T, *, max_chars: int | None = None) -> str: ...
    def to_json(self, value: T) -> Any: ...
    def from_json(self, data: Any) -> T: ...
    def json_schema(self) -> dict[str, Any]: ...
    def fingerprint(self, value: T) -> str: ...


def _truncate(text: str, max_chars: int | None) -> str:
    if max_chars is None or len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 1)] + "…"


class TypeAdapterCodec(Generic[T]):
    """Covers Pydantic models, dataclasses, TypedDicts, dict/list/str — anything Pydantic
    can validate. This is the default path for 'arbitrary data'."""

    def __init__(self, tp: Any) -> None:
        self._ta: TypeAdapter[T] = TypeAdapter(tp)

    def to_json(self, value: T) -> Any:
        return self._ta.dump_python(value, mode="json")

    def from_json(self, data: Any) -> T:
        return self._ta.validate_python(data)

    def to_text(self, value: T, *, max_chars: int | None = None) -> str:
        data = self.to_json(value)
        text = data if isinstance(data, str) else json.dumps(data, indent=2, sort_keys=True)
        return _truncate(text, max_chars)

    def json_schema(self) -> dict[str, Any]:
        return self._ta.json_schema()

    def fingerprint(self, value: T) -> str:
        return fingerprint_json(self.to_json(value))


class FunctionCodec(Generic[T]):
    """Escape hatch: wrap your own render/serialize functions (DataFrame rows, images, ...)."""

    def __init__(self, *, to_text, to_json, from_json, schema: dict[str, Any] | None = None):
        self._t, self._j, self._f, self._s = to_text, to_json, from_json, schema or {}

    def to_text(self, value: T, *, max_chars: int | None = None) -> str:
        return _truncate(self._t(value), max_chars)

    def to_json(self, value: T) -> Any:
        return self._j(value)

    def from_json(self, data: Any) -> T:
        return self._f(data)

    def json_schema(self) -> dict[str, Any]:
        return self._s

    def fingerprint(self, value: T) -> str:
        return fingerprint_json(self._j(value))


class CodecRegistry:
    def __init__(self) -> None:
        self._by_kind: dict[str, Codec[Any]] = {}

    def register(self, kind: str, codec_or_type: Any) -> None:
        is_codec = isinstance(codec_or_type, (TypeAdapterCodec, FunctionCodec)) or (
            not isinstance(codec_or_type, type) and isinstance(codec_or_type, Codec)
        )
        self._by_kind[kind] = (
            cast("Codec[Any]", codec_or_type) if is_codec else TypeAdapterCodec(codec_or_type)
        )

    def get(self, kind: str) -> Codec[Any]:
        try:
            return self._by_kind[kind]
        except KeyError:
            raise KeyError(f"no codec registered for kind {kind!r}") from None

    def wrap(self, kind: str, payload: Any, **kw: Any) -> Item[Any]:
        """Create an Item and stamp its content hash via the registered codec."""
        return Item(
            kind=kind, payload=payload, content_hash=self.get(kind).fingerprint(payload), **kw
        )

    def render(self, item: Item[Any], *, max_chars: int | None = None) -> str:
        return self.get(item.kind).to_text(item.payload, max_chars=max_chars)
