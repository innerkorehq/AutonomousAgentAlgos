"""Trace sinks: where traces go.

A sink receives complete, ordered batches of `Span`s (parents before children) and ships them somewhere. The only
required method is `export`; scoring is an optional capability (`aaa.score.Scorable`), because not every backend
can attach scores to a trace."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from aaa.score import Scorable
from aaa.trace import Span


class ExportError(Exception):
    """A sink could not deliver a batch of spans."""


@runtime_checkable
class TraceSink(Protocol):
    async def export(self, spans: Sequence[Span]) -> None:
        """Deliver `spans` (ordered parents first). Raise `ExportError` on failure; the observer logs it and
        carries on, so a broken sink never affects a run. Must be idempotent: ids are deterministic and
        the same span may be sent again after a restart."""
        ...

    async def aclose(self) -> None: ...


__all__ = ["ExportError", "Scorable", "TraceSink"]
