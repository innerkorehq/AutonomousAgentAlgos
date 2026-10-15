"""Sinks that need no third-party library: for tests, local debugging and wrapping your own exporter."""

from __future__ import annotations

import inspect
import sys
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, TextIO

from aaa.score import Score
from aaa.trace import Span


class InMemorySink:
    """Keeps everything it is given. The reference sink (and the one tests assert against)."""

    def __init__(self) -> None:
        self.spans: list[Span] = []
        self.scores: list[tuple[str, str | None, Score]] = []  # (trace id, span id, score)
        self.batches = 0

    async def export(self, spans: Sequence[Span]) -> None:
        self.batches += 1
        self.spans.extend(spans)

    async def score(
        self, trace_id: str, scores: Sequence[Score], *, span_id: str | None = None
    ) -> None:
        self.scores += [(trace_id, span_id, s) for s in scores]

    def by_trace(self) -> dict[str, list[Span]]:
        out: dict[str, list[Span]] = {}
        for s in self.spans:
            out.setdefault(s.trace_id, []).append(s)
        return out

    async def aclose(self) -> None:
        return None


class ConsoleSink:
    """Prints each trace as an indented tree: the quickest way to see what a run did."""

    def __init__(self, out: TextIO | None = None) -> None:
        self.out = out

    async def export(self, spans: Sequence[Span]) -> None:
        out = self.out or sys.stdout
        children: dict[str | None, list[Span]] = {}
        ids = {s.span_id for s in spans}
        for s in spans:
            children.setdefault(s.parent_id if s.parent_id in ids else None, []).append(s)

        def show(s: Span, depth: int) -> None:
            dur = f"{s.duration_s * 1000:.0f}ms" if s.duration_s is not None else "open"
            extra = []
            if (tok := s.attributes.get("usage.output_tokens")) is not None:
                extra.append(f"{s.attributes.get('usage.input_tokens', 0)}+{tok} tok")
            if (c := s.attributes.get("cost")) is not None:
                extra.append(f"${c:.4f}")
            mark = {"ok": "", "error": " ERROR", "incomplete": " (incomplete)"}[s.status]
            print(
                f"{'  ' * depth}{s.name} [{dur}{', ' + ', '.join(extra) if extra else ''}]{mark}",
                file=out,
            )
            for c in children.get(s.span_id, []):
                show(c, depth + 1)

        for root in children.get(None, []):
            show(root, 0)

    async def aclose(self) -> None:
        return None


class CallableSink:
    """Wrap a function `f(spans)` (sync or async) as a sink: the shortest way to ship traces to anything."""

    def __init__(self, export: Callable[[Sequence[Span]], Any | Awaitable[Any]]) -> None:
        self._export = export

    async def export(self, spans: Sequence[Span]) -> None:
        r = self._export(spans)
        if inspect.isawaitable(r):
            await r

    async def aclose(self) -> None:
        return None
