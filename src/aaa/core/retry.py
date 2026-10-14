"""Bounded retry for infrastructure failures (a store that is briefly unreachable).

Retrying is safe here because every append is idempotent on event id: repeating one that actually
succeeded is absorbed by the store, so "did it commit before the connection died?" never matters."""

from __future__ import annotations

import asyncio
import logging
import random
import sqlite3
import time
from collections.abc import Awaitable, Callable
from typing import TypeVar

from pydantic import BaseModel, ConfigDict, Field

from aaa.core.errors import ConcurrencyError, SchemaError, StoreUnavailable

log = logging.getLogger("aaa.retry")
T = TypeVar("T")


class RetryPolicy(BaseModel):
    """How long to keep trying before giving up on an infrastructure operation.

    Retrying stops at the FIRST bound reached; a bound set to `None` is not applied, and a policy
    with both bounds `None` retries until the call succeeds (or the task is cancelled)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_attempts: int | None = Field(None, ge=1)
    """Total tries including the first."""
    max_elapsed_s: float | None = Field(30.0, gt=0)
    """Total time budget for backing off (not counting the time the operation itself takes)."""
    base_delay_s: float = Field(0.1, gt=0)
    max_delay_s: float = Field(2.0, gt=0)
    multiplier: float = Field(2.0, ge=1)
    jitter: bool = True
    """Full jitter (uniform in [0, delay]) so many workers do not retry in lock-step."""

    def delay(self, attempt: int) -> float:
        """Backoff before retry number `attempt` (1-based)."""
        d = min(self.max_delay_s, self.base_delay_s * self.multiplier ** (attempt - 1))
        return random.uniform(0, d) if self.jitter else d

    @classmethod
    def forever(cls, base_delay_s: float = 0.2, max_delay_s: float = 5.0) -> RetryPolicy:
        """Never give up: for results that must not be lost (a tool that already ran)."""
        return cls(
            max_attempts=None,
            max_elapsed_s=None,
            base_delay_s=base_delay_s,
            max_delay_s=max_delay_s,
        )


# Exception classes (matched by name through the MRO, so optional drivers are never imported) that
# mean "the connection or server is unavailable right now". Observed from real servers: Postgres
# reports a killed connection as `InternalClientError` (!), Redis as `ConnectionError`.
_TRANSIENT_NAMES = frozenset(
    {
        # asyncpg
        "PostgresConnectionError",
        "ConnectionDoesNotExistError",
        "ConnectionFailureError",
        "CannotConnectNowError",
        "TooManyConnectionsError",
        "AdminShutdownError",
        "CrashShutdownError",
        "InterfaceError",
        "InternalClientError",
        "QueryCanceledError",
        "DeadlockDetectedError",
        "SerializationError",
        # redis-py
        "ConnectionError",
        "TimeoutError",
        "BusyLoadingError",
        "ClusterDownError",
        "TryAgainError",
    }
)
_NOT_TRANSIENT_OS = (FileNotFoundError, PermissionError, IsADirectoryError, NotADirectoryError)
_SQLITE_TRANSIENT = ("locked", "busy", "disk i/o")


def is_transient(exc: BaseException) -> bool:
    """True if `exc` looks like a temporary infrastructure problem worth retrying.

    Deliberately conservative: logical errors (`ConcurrencyError`, `SchemaError`, validation errors,
    bugs) are never transient, so a genuine fault still fails fast instead of looping."""
    if isinstance(exc, StoreUnavailable):
        return True
    if isinstance(exc, ConcurrencyError | SchemaError | asyncio.CancelledError):
        return False
    if isinstance(exc, _NOT_TRANSIENT_OS):
        return False
    if isinstance(
        exc, OSError | TimeoutError
    ):  # ConnectionError, ConnectionResetError, ... included
        return True
    if isinstance(exc, sqlite3.OperationalError):
        return any(s in str(exc).lower() for s in _SQLITE_TRANSIENT)
    for cls in type(exc).__mro__:
        if cls.__name__ in _TRANSIENT_NAMES and cls.__module__.split(".")[0] in {
            "asyncpg",
            "redis",
        }:
            return True
    return False


async def retrying(
    fn: Callable[[], Awaitable[T]],
    policy: RetryPolicy,
    *,
    classify: Callable[[BaseException], bool] = is_transient,
    what: str = "store operation",
) -> T:
    """Run `fn`, retrying transient failures per `policy`. Non-transient errors propagate untouched;
    exhausting the policy raises `StoreUnavailable` (chained to the last error)."""
    attempt, started = 0, time.monotonic()
    while True:
        attempt += 1
        try:
            return await fn()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if not classify(exc):
                raise
            delay = policy.delay(attempt)
            spent = time.monotonic() - started
            out_of_attempts = policy.max_attempts is not None and attempt >= policy.max_attempts
            out_of_time = policy.max_elapsed_s is not None and spent + delay > policy.max_elapsed_s
            if out_of_attempts or out_of_time:
                raise StoreUnavailable(
                    f"{what} still failing after {attempt} attempt(s) over {spent:.1f}s: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            level = logging.WARNING if attempt == 1 or attempt % 10 == 0 else logging.DEBUG
            log.log(
                level,
                "%s failed (%s: %s); retry %d in %.2fs",
                what,
                type(exc).__name__,
                exc,
                attempt,
                delay,
            )
            await asyncio.sleep(delay)
