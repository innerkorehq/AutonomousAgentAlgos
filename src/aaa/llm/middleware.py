from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable, Callable

from aaa.llm.client import LLMClient
from aaa.llm.errors import CircuitOpen, LLMError, RateLimited, TransientError
from aaa.llm.types import EmbedResponse, LLMRequest, LLMResponse


class _Wrapper:
    def __init__(self, inner: LLMClient) -> None:
        self.inner = inner

    def __getattr__(self, name: str):
        # Streaming passes straight through (no retry/breaker once bytes may have been seen);
        # absent on the inner client => absent here, so capability checks stay truthful.
        if name == "stream":
            return getattr(self.inner, "stream")  # noqa: B009 - must raise AttributeError when absent
        raise AttributeError(name)

    async def embed(self, texts: list[str], model: str | None = None) -> EmbedResponse:
        return await self._call(lambda: self.inner.embed(texts, model))

    async def complete(self, request: LLMRequest) -> LLMResponse:
        return await self._call(lambda: self.inner.complete(request))

    async def aclose(self) -> None:
        await self.inner.aclose()

    async def _call(self, fn: Callable[[], Awaitable]):  # pragma: no cover
        raise NotImplementedError


class RetryingClient(_Wrapper):
    """Bounded exponential backoff with full jitter on retryable errors only. Honors a
    gateway's Retry-After. Use only one retry layer: if your gateway already retries,
    set max_attempts=1."""

    def __init__(
        self,
        inner: LLMClient,
        *,
        max_attempts: int = 3,
        base_delay: float = 0.5,
        max_delay: float = 20.0,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        super().__init__(inner)
        self.max_attempts, self.base, self.cap, self._sleep = (
            max_attempts,
            base_delay,
            max_delay,
            sleep,
        )

    async def _call(self, fn):
        for attempt in range(1, self.max_attempts + 1):
            try:
                return await fn()
            except LLMError as e:
                if not e.retryable or attempt == self.max_attempts:
                    raise
                delay = random.uniform(0, min(self.cap, self.base * 2 ** (attempt - 1)))
                if isinstance(e, RateLimited) and e.retry_after:
                    delay = max(delay, min(e.retry_after, self.cap))
                await self._sleep(delay)
        raise AssertionError("unreachable")


class CircuitBreakerClient(_Wrapper):
    """closed -> (N consecutive transient failures) -> open -> (cooldown) -> half-open probe.
    Only transient/rate-limit failures count; caller mistakes never trip it."""

    def __init__(
        self,
        inner: LLMClient,
        *,
        failure_threshold: int = 5,
        cooldown_s: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__(inner)
        self.threshold, self.cooldown, self._clock = failure_threshold, cooldown_s, clock
        self._failures = 0
        self._opened_at: float | None = None
        self._probing = False

    @property
    def state(self) -> str:
        if self._opened_at is None:
            return "closed"
        return "half_open" if self._clock() - self._opened_at >= self.cooldown else "open"

    async def _call(self, fn):
        st = self.state
        if st == "open" or (st == "half_open" and self._probing):
            raise CircuitOpen("circuit open")
        if st == "half_open":
            self._probing = True
        try:
            result = await fn()
        except TransientError:
            self._failures += 1
            if self._failures >= self.threshold or st == "half_open":
                self._opened_at = self._clock()
            raise
        finally:
            self._probing = False
        self._failures, self._opened_at = 0, None
        return result
