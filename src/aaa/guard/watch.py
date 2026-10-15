"""Observe-mode guards: run out of line, record verdicts, never change the run.

    GuardWatcher(engine, GuardPolicy([GuardSpec(LLMJudgeGuard(...), mode="observe")])).attach()

Use it to measure a new guard on real traffic before enforcing it. The watcher reads events the engine already carries, so a slow or
failing guard cannot delay or fail the watched run (see `aaa.safe`)."""

from __future__ import annotations

import json
from typing import Self

from aaa.events import Event
from aaa.guard.enforce import Enforcer
from aaa.guard.policy import GuardPolicy
from aaa.guard.types import GuardContext
from aaa.llm.types import LLMResponse
from aaa.runtime import Engine
from aaa.safe import SafeWorker, safe_subscribe


class GuardWatcher:
    def __init__(
        self, engine: Engine, policy: GuardPolicy, *, group: str = "aaa.guard.watch"
    ) -> None:
        self.engine, self.policy, self.group = engine, policy, group
        self.enforcer = Enforcer(engine, policy)
        self._workers: list[SafeWorker] = []

    def attach(self) -> Self:
        if not self._workers:
            for pattern, fn in (
                ("aaa.llm.requested", self._input),
                ("aaa.llm.completed", self._output),
                ("aaa.tool.requested", self._tool_call),
                ("aaa.tool.completed", self._tool_result),
            ):
                self._workers.append(
                    safe_subscribe(
                        self.engine,
                        pattern,
                        fn,
                        name=f"aaa.guard.watch:{pattern}",
                        group=f"{self.group}:{pattern}",
                    )
                )
        return self

    async def __aenter__(self) -> Self:
        return self.attach()

    async def __aexit__(self, *exc: object) -> None:
        await self.drain()
        await self.stop()

    async def drain(self, timeout_s: float = 30.0) -> bool:
        await self.engine.bus.drain()
        return all([await w.drain(timeout_s) for w in self._workers])

    async def stop(self) -> None:
        for w in self._workers:
            await w.stop()
        self._workers = []

    def _ctx(self, stage, e: Event, **kw) -> GuardContext:  # type: ignore[no-untyped-def]
        return GuardContext(stage, e.run_id, e.tenant, dict(e.meta), **kw)

    async def _input(self, e: Event) -> None:
        req = (await self.engine.hydrate(e)).data["request"]
        for m in req.get("messages", []):
            if m.get("role") == "user" and m.get("content"):
                await self.enforcer.decide(
                    "input", m["content"], self._ctx("input", e), e, mode="observe"
                )

    async def _output(self, e: Event) -> None:
        content = LLMResponse.model_validate(
            (await self.engine.hydrate(e)).data["response"]
        ).message.content
        if content:
            await self.enforcer.decide("output", content, self._ctx("output", e), e, mode="observe")

    async def _tool_call(self, e: Event) -> None:
        args = (await self.engine.hydrate(e)).data.get("arguments") or {}
        ctx = self._ctx("tool_call", e, tool=e.data.get("name"), arguments=args)
        await self.enforcer.decide(
            "tool_call", json.dumps(args, default=str, sort_keys=True), ctx, e, mode="observe"
        )

    async def _tool_result(self, e: Event) -> None:
        r = (await self.engine.hydrate(e)).data.get("result")
        text = r if isinstance(r, str) else json.dumps(r, default=str, sort_keys=True)
        await self.enforcer.decide(
            "tool_result",
            text,
            self._ctx("tool_result", e, tool=e.data.get("name")),
            e,
            mode="observe",
        )
