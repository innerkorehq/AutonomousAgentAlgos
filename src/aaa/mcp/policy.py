"""Per-tool rules a server enforces at call time: visibility (scopes, tenants), a timeout, a rate limit and a concurrency cap.

Every refusal is deliberately plain. A tool the caller may not see answers exactly like a tool that does not exist; rate-limit and
concurrency refusals say only that the caller should retry, never how the limits are set."""

from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class ToolRule:
    timeout_s: float | None = None
    rate_per_minute: int | None = None
    burst: int | None = None
    max_concurrency: int | None = None
    scopes: frozenset[str] | None = None
    tenants: frozenset[str] | None = None

    def visible_to(self, tenant: str, scopes: frozenset[str] | None) -> bool:
        """`scopes=None` is a caller the host did not restrict. Otherwise the caller needs one of the tool's scopes."""
        if self.tenants is not None and tenant not in self.tenants:
            return False
        return not (self.scopes is not None and scopes is not None and not (self.scopes & scopes))


class Limiter:
    """Token buckets keyed by (caller, tool), bounded in memory (the least recently used keys are forgotten) and with an injectable clock."""

    def __init__(self, clock: Callable[[], float] | None = None, max_keys: int = 10_000) -> None:
        self.clock = clock or time.monotonic
        self.max_keys = max_keys
        self._buckets: OrderedDict[str, tuple[float, float]] = (
            OrderedDict()
        )  # key -> (tokens, last)

    def allow(self, key: str, per_minute: int, burst: int | None) -> bool:
        capacity = float(burst or min(per_minute, 10))
        rate = per_minute / 60.0
        now = self.clock()
        tokens, last = self._buckets.pop(key, (capacity, now))
        tokens = min(capacity, tokens + max(0.0, now - last) * rate)
        ok = tokens >= 1.0
        self._buckets[key] = (tokens - 1.0 if ok else tokens, now)
        while len(self._buckets) > self.max_keys:
            self._buckets.popitem(last=False)
        return ok


class Gates:
    """A cap on calls in flight per tool; `enter` refuses at once when the tool is at its cap instead of queueing."""

    def __init__(self) -> None:
        self._active: dict[str, int] = {}

    def enter(self, name: str, limit: int) -> bool:
        if self._active.get(name, 0) >= limit:
            return False
        self._active[name] = self._active.get(name, 0) + 1
        return True

    def leave(self, name: str) -> None:
        self._active[name] = max(0, self._active.get(name, 0) - 1)
