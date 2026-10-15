"""Admission control: rate limits, concurrency caps and event-sourced usage quotas, all keyed by tenant.

Decisions are made BEFORE any work is started. Quota is a projection over usage events in the same event store
that holds the runs, so it survives restarts and is the same on every server process; the check-and-reserve is an
optimistic-concurrency append, so two requests racing for the last unit of quota cannot both win."""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic import BaseModel

from aaa.core.errors import ConcurrencyError
from aaa.events import Event, Projection

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from aaa.events import EventStore


class Denied(Exception):
    """A request was refused before any work started. `status` is the HTTP status to answer with."""

    def __init__(self, status: int, code: str, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.status, self.code, self.message, self.retry_after = status, code, message, retry_after


# -- rate limiting -------------------------------------------------------------------------------
@dataclass(frozen=True)
class RateDecision:
    allowed: bool
    retry_after: float = 0.0  # seconds until `cost` tokens will be available


@runtime_checkable
class RateLimiter(Protocol):
    async def take(self, key: str, cost: float = 1.0) -> RateDecision: ...


class InMemoryRateLimiter:
    """Token bucket per key: `burst` tokens, refilled at `per_second`. One process only; use the Redis limiter to
    share a limit between server processes."""

    def __init__(
        self, per_second: float, burst: float, *, clock: Callable[[], float] | None = None
    ) -> None:
        if per_second <= 0 or burst < 1:
            raise ValueError("per_second must be > 0 and burst >= 1")
        self.rate, self.burst = per_second, burst
        self._clock = clock or time.monotonic
        self._state: dict[str, tuple[float, float]] = {}  # key -> (tokens, last)

    async def take(self, key: str, cost: float = 1.0) -> RateDecision:
        now = self._clock()
        tokens, last = self._state.get(key, (self.burst, now))
        tokens = min(self.burst, tokens + (now - last) * self.rate)
        if tokens >= cost:
            self._state[key] = (tokens - cost, now)
            return RateDecision(True)
        self._state[key] = (tokens, now)
        return RateDecision(False, (cost - tokens) / self.rate)


_LUA_BUCKET = """
local t = redis.call('TIME')
local now = tonumber(t[1]) + tonumber(t[2]) / 1000000
local rate, burst, cost = tonumber(ARGV[1]), tonumber(ARGV[2]), tonumber(ARGV[3])
local s = redis.call('HMGET', KEYS[1], 'tokens', 'ts')
local tokens, ts = tonumber(s[1]), tonumber(s[2])
if tokens == nil then tokens, ts = burst, now end
tokens = math.min(burst, tokens + math.max(0, now - ts) * rate)
local ok, wait = 0, 0
if tokens >= cost then tokens = tokens - cost; ok = 1 else wait = (cost - tokens) / rate end
redis.call('HSET', KEYS[1], 'tokens', tostring(tokens), 'ts', tostring(now))
redis.call('PEXPIRE', KEYS[1], math.ceil(burst / rate * 1000) + 1000)
return {ok, tostring(wait)}
"""


class RedisRateLimiter:
    """The same token bucket in Redis, evaluated atomically by a Lua script, so every server process shares one
    limit. Time comes from the Redis server (`TIME`), not from the callers' clocks."""

    def __init__(self, redis: Any, per_second: float, burst: float, *, prefix: str = "aaa:rl:"):
        if per_second <= 0 or burst < 1:
            raise ValueError("per_second must be > 0 and burst >= 1")
        self.redis, self.rate, self.burst, self.prefix = redis, per_second, burst, prefix

    async def take(self, key: str, cost: float = 1.0) -> RateDecision:
        ok, wait = await self.redis.eval(
            _LUA_BUCKET, 1, self.prefix + key, self.rate, self.burst, cost
        )
        return RateDecision(bool(int(ok)), float(wait))


# -- concurrency ---------------------------------------------------------------------------------
class ConcurrencyLimiter:
    """At most `limit` things in flight per key in THIS process (used for open SSE streams)."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self._n: dict[str, int] = defaultdict(int)

    @asynccontextmanager
    async def hold(self, key: str) -> AsyncIterator[None]:
        if self._n[key] >= self.limit:
            raise Denied(429, "too_many_streams", "too many concurrent streams", retry_after=1.0)
        self._n[key] += 1
        try:
            yield
        finally:
            self._n[key] -= 1
            if self._n[key] <= 0:
                self._n.pop(key, None)

    def active(self, key: str) -> int:
        return self._n.get(key, 0)


# -- quota ---------------------------------------------------------------------------------------
ADMITTED = "aaa_server.usage.admitted"
SETTLED = "aaa_server.usage.settled"


class Usage(BaseModel):
    admitted: int = 0
    settled: int = 0
    tokens: int = 0
    cost: float = 0.0


def _fold(u: Usage, e: Event) -> Usage:
    if e.type == ADMITTED:
        u.admitted += 1
    elif e.type == SETTLED:
        u.settled += 1
        u.tokens += int(e.data.get("tokens", 0))
        u.cost += float(e.data.get("cost", 0.0))
    return u


USAGE = Projection(Usage, _fold, name="aaa_server.usage", version=1, state_type=Usage, every=256)


class Quota(BaseModel):
    """Per-tenant limits per UTC day. `None` = unlimited. `max_cost` is checked against spend SETTLED so far
    (a run's cost is recorded when it finishes), so concurrent runs can overshoot it by what they are still
    spending; pair it with per-run budgets."""

    max_runs: int | None = None
    max_tokens: int | None = None
    max_cost: float | None = None


class QuotaStore:
    def __init__(
        self,
        store: EventStore,
        default: Quota,
        *,
        overrides: dict[str, Quota] | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.store, self.default, self.overrides = store, default, overrides or {}
        self._clock = clock or (lambda: datetime.now(UTC))

    def limits(self, tenant: str) -> Quota:
        return self.overrides.get(tenant, self.default)

    def stream(self, tenant: str) -> str:
        return f"aaa_server.quota/{tenant}/{self._clock():%Y-%m-%d}"

    async def usage(self, tenant: str) -> Usage:
        return (await USAGE.load(self.store, self.stream(tenant))).state

    async def admit(self, tenant: str, unit_id: str) -> None:
        """Reserve one run/turn against today's quota or raise `Denied(429)`. Idempotent per `unit_id`."""
        lim = self.limits(tenant)
        stream = self.stream(tenant)
        for _ in range(8):
            events = await self.store.read(stream)
            u = USAGE.replay(events)
            if any(e.id == f"{unit_id}:admit" for e in events):
                return
            for cap, used, what in (
                (lim.max_runs, u.admitted, "runs"),
                (lim.max_tokens, u.tokens, "tokens"),
                (lim.max_cost, u.cost, "cost"),
            ):
                if cap is not None and used >= cap:
                    raise Denied(429, "quota_exceeded", f"daily {what} quota exhausted")
            try:
                await self.store.append(
                    stream,
                    [
                        Event(
                            id=f"{unit_id}:admit",
                            type=ADMITTED,
                            stream=stream,
                            tenant=tenant,
                            source="aaa-server",
                        )
                    ],
                    expected_seq=events[-1].seq if events else -1,
                )
                return
            except ConcurrencyError:
                continue  # someone else took a unit: re-check against the new total
        raise Denied(503, "busy", "could not reserve quota, retry", retry_after=1.0)

    async def settle(self, tenant: str, unit_id: str, *, tokens: int, cost: float) -> None:
        """Record what a finished run actually used (idempotent per `unit_id`)."""
        stream = self.stream(tenant)
        await self.store.append(
            stream,
            [
                Event(
                    id=f"{unit_id}:settle",
                    type=SETTLED,
                    stream=stream,
                    tenant=tenant,
                    source="aaa-server",
                    data={"tokens": tokens, "cost": cost},
                )
            ],
        )


__all__ = [
    "ConcurrencyLimiter",
    "Denied",
    "InMemoryRateLimiter",
    "Quota",
    "QuotaStore",
    "RateDecision",
    "RateLimiter",
    "RedisRateLimiter",
    "Usage",
]
