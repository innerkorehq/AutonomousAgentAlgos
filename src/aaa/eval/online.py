"""Online evaluation: score a sample of real runs as they finish, and record the scores on the run.

    async with OnlineEvaluator(engine, [LLMJudge(client, "Is the answer polite?")], sample=0.05, sinks=[langfuse]):
        ...  # production traffic

* Runs in its own competing-consumer group, outside the scored run: it can be slow or fail without touching the run.
* Samples deterministically by trace id, so every process picks the same runs.
* Never scores eval runs or judge runs (no feedback loops): runs stamped `aaa_eval` are skipped.
* `max_cost` caps what judges may spend in this process; past it, runs are skipped and counted.
* Each score is appended to the run's log as `aaa.eval.score` (idempotent: the id is derived from the run and metric), so it
  shows up as an evaluator span in later exports, and is sent to any `Scorable` sink (Langfuse, ...) for the run's trace."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, Self

from aaa.core.outcome import Outcome
from aaa.eval.dataset import Case
from aaa.eval.metrics import Metric
from aaa.eval.result import CaseResult, default_extract
from aaa.eval.suite import EVAL_META, _score, fill_trajectory
from aaa.events import Event
from aaa.runtime import Engine
from aaa.safe import SafeWorker, safe_subscribe
from aaa.score import Score
from aaa.trace import span_id_of, trace_id_of

log = logging.getLogger("aaa.eval")


class OnlineEvaluator:
    def __init__(
        self,
        engine: Engine,
        metrics: Sequence[Metric],
        *,
        sample: float = 1.0,
        sinks: Sequence[Any] = (),
        max_cost: float | None = None,
        tenant: str | None = None,
        group: str = "aaa.eval",
    ) -> None:
        self.engine, self.metrics, self.sample, self.sinks = (
            engine,
            list(metrics),
            sample,
            list(sinks),
        )
        self.max_cost, self.tenant, self.group = max_cost, tenant, group
        self.spent = 0.0
        self.evaluated = 0
        self.skipped_over_budget = 0
        self._worker: SafeWorker | None = None

    def attach(self) -> Self:
        if self._worker is None:
            self._worker = safe_subscribe(
                self.engine,
                "aaa.run.completed",
                self._on_completed,
                name="aaa.eval",
                group=self.group,
            )
        return self

    async def __aenter__(self) -> Self:
        return self.attach()

    async def __aexit__(self, *exc: object) -> None:
        await self.drain()
        await self.stop()

    async def drain(self, timeout_s: float = 30.0) -> bool:
        if self._worker is None:
            return True
        await self.engine.bus.drain()
        return await self._worker.drain(timeout_s)

    async def stop(self) -> None:
        if self._worker is not None:
            await self._worker.stop()
            self._worker = None

    def _sampled(self, trace_id: str) -> bool:
        return self.sample >= 1.0 or (int(trace_id[:8], 16) % 10_000) < self.sample * 10_000

    async def _on_completed(self, e: Event) -> None:
        run_id = e.run_id
        if not run_id:
            return
        eng = self.engine
        head = await eng.retry(lambda: eng.store.read(run_id, 0, 1), "read")
        if not head:
            return
        started = head[0]
        if started.data.get("parent") or started.meta.get(EVAL_META):
            return
        if self.tenant is not None and started.tenant != self.tenant:
            return
        trace_id = trace_id_of(run_id)
        if not self._sampled(trace_id):
            return
        if self.max_cost is not None and self.spent >= self.max_cost:
            self.skipped_over_budget += 1
            return
        outcome = Outcome(**e.data)
        res = CaseResult(
            case=Case(id=run_id, input=dict(started.data.get("input") or {})),
            outcome=outcome,
            text=default_extract(outcome.value),
            run_id=run_id,
        )
        await fill_trajectory(eng, res)
        for m in self.metrics:
            res.scores.extend(await _score(m, res))
        self.spent += sum(s.cost for s in res.scores)
        self.evaluated += 1
        for s in res.scores:
            await self._record(started, e, s)
        for sink in self.sinks:
            if hasattr(sink, "score"):
                try:
                    await sink.score(trace_id, res.scores, span_id=span_id_of("run", run_id))
                except Exception:
                    log.warning("score sink %s failed", type(sink).__name__, exc_info=True)

    async def _record(self, started: Event, completed: Event, s: Score) -> None:
        data = {
            "name": s.name,
            "value": s.value,
            "reason": s.reason,
            "usage": s.usage,
            "cost": s.cost,
        }
        ev = completed.caused("aaa.eval.score", data, meta={**started.meta, EVAL_META: True})
        eng = self.engine
        await eng.retry(lambda: eng.store.append(ev.stream, [ev]), "append")
