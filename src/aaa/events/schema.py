from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ValidationError

from aaa.core.errors import SchemaError
from aaa.events.event import Event

Upcaster = Callable[[dict[str, Any]], dict[str, Any]]


def _split(ref: str) -> tuple[str, int]:
    name, _, ver = ref.rpartition("/v")
    if not name or not ver.isdigit():
        raise SchemaError(f"bad schema ref {ref!r}; expected '<name>/v<N>'")
    return name, int(ver)


class SchemaRegistry:
    """Versioned payload schemas with upcasters that migrate old events on read."""

    def __init__(self) -> None:
        self._models: dict[str, type[BaseModel]] = {}
        self._upcasters: dict[tuple[str, int], Upcaster] = {}  # (name, from_version)->v+1
        self._latest: dict[str, int] = {}

    def register(self, ref: str, model: type[BaseModel]) -> None:
        name, ver = _split(ref)
        self._models[ref] = model
        self._latest[name] = max(self._latest.get(name, 0), ver)

    def upcaster(self, name: str, from_version: int, fn: Upcaster) -> None:
        self._upcasters[(name, from_version)] = fn

    def validate(self, event: Event) -> Event:
        """Return the event upcast to the latest version and validated; unknown refs pass."""
        if not event.schema_ref:
            return event
        name, ver = _split(event.schema_ref)
        if name not in self._latest:
            return event
        data = event.data
        while ver < self._latest[name]:
            fn = self._upcasters.get((name, ver))
            if fn is None:
                raise SchemaError(f"no upcaster for {name}/v{ver}")
            data, ver = fn(data), ver + 1
        model = self._models[f"{name}/v{ver}"]
        try:
            data = model.model_validate(data).model_dump(mode="json")
        except ValidationError as e:
            raise SchemaError(f"{event.schema_ref}: {e}") from e
        return event.model_copy(update={"data": data, "schema_ref": f"{name}/v{ver}"})
