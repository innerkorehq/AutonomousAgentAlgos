from __future__ import annotations

import inspect
import json
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, TypeAdapter, ValidationError

from aaa.events import Event
from aaa.runtime import Engine

REQUESTED = "aaa.validate.requested"
COMPLETED = "aaa.validate.completed"


class Verdict(BaseModel):
    ok: bool
    feedback: str = ""
    score: float | None = None


def _coerce(r: Any) -> Verdict:
    if isinstance(r, Verdict):
        return r
    if isinstance(r, bool):
        return Verdict(ok=r, feedback="" if r else "validation failed")
    if isinstance(r, tuple):
        return Verdict(ok=bool(r[0]), feedback=str(r[1]))
    raise TypeError(f"validator must return Verdict | bool | (bool, str), got {type(r).__name__}")


def json_validator(tp: Any) -> Callable[[str, dict], Verdict]:
    """Deterministic check that a candidate is JSON matching any pydantic-validatable type."""
    ta = TypeAdapter(tp)

    def check(candidate: str, context: dict) -> Verdict:
        try:
            ta.validate_python(json.loads(candidate))
        except ValueError as e:
            if isinstance(e, ValidationError):
                return Verdict(
                    ok=False,
                    feedback="; ".join(
                        f"{'.'.join(map(str, x['loc']))}: {x['msg']}" for x in e.errors()
                    ),
                )
            return Verdict(ok=False, feedback=f"not valid JSON: {e}")
        return Verdict(ok=True)

    return check


class ValidationHandler:
    """Runs named, deterministic validators: `aaa.validate.requested` {candidate, validator,
    context} -> `aaa.validate.completed` {verdict}. The 'test' half of generate-test-repair;
    validators are plain (async) callables — unit tests, linters, schema checks, business rules."""

    def __init__(self, validators: dict[str, Callable[..., Any]]) -> None:
        self.validators = validators

    def attach(self, engine: Engine) -> None:
        engine.subscribe(REQUESTED, self.handle, group="aaa.validate")

    async def handle(self, e: Event) -> list[Event]:
        name = e.data.get("validator", "default")
        fn = self.validators.get(name)
        if fn is None:
            v = Verdict(ok=False, feedback=f"no validator named {name!r}")
        else:
            try:
                r = fn(e.data["candidate"], e.data.get("context") or {})
                v = _coerce(await r if inspect.isawaitable(r) else r)
            except Exception as exc:  # noqa: BLE001 - a crashing validator is a failed validation
                v = Verdict(ok=False, feedback=f"validator error: {type(exc).__name__}: {exc}")
        return [e.caused(COMPLETED, {"verdict": v.model_dump()}, id=f"{e.id}:result")]


__all__ = ["ValidationHandler", "Verdict", "json_validator"]
