"""Run a dataset through an algorithm and score the results."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Sequence
from typing import Any

from aaa.eval.dataset import Case, Dataset
from aaa.eval.metrics import Metric
from aaa.eval.report import EvalReport
from aaa.eval.result import CaseResult, default_extract
from aaa.runtime import Budget, Engine
from aaa.runtime.engine import RUNMETA, Algorithm
from aaa.score import Score
from aaa.trace import children_of, spans_from_events

log = logging.getLogger("aaa.eval")
EVAL_META = "aaa_eval"  # stamped on eval runs so online evaluators never score them


class EvalSuite:
    """`EvalSuite(dataset, algo, [metrics]).run(engine)` -> `EvalReport`.

    * Each case is a real run (`engine.run`) with `meta={"aaa_eval": True}`; the trajectory (spans), steps, tokens and cost are read
      back from its log, so tool-use and budget metrics see what really happened.
    * `repeats` runs each case several times (agents are non-deterministic); the report keeps every run.
    * A metric that raises is recorded as a failed score, not a crashed suite; a run the harness could not start is `error`.
    * `budget` bounds each case, so one runaway case cannot spend the whole eval."""

    def __init__(
        self,
        dataset: Dataset,
        algo: Algorithm,
        metrics: Sequence[Metric],
        *,
        name: str | None = None,
        concurrency: int = 4,
        repeats: int = 1,
        budget: Budget | None = None,
        tenant: str | None = None,
        extract: Callable[[Any], str] = default_extract,
        make_input: Callable[[Case], dict[str, Any]] | None = None,
    ) -> None:
        self.dataset, self.algo, self.metrics = dataset, algo, list(metrics)
        self.name = name or dataset.name
        self.concurrency, self.repeats, self.budget, self.tenant = (
            max(1, concurrency),
            max(1, repeats),
            budget,
            tenant,
        )
        self.extract, self.make_input = extract, make_input

    async def run(self, engine: Engine) -> EvalReport:
        sem = asyncio.Semaphore(self.concurrency)

        async def one(case: Case, rep: int) -> CaseResult:
            async with sem:
                return await self._run_case(engine, case, rep)

        jobs = [one(c, r) for c in self.dataset for r in range(self.repeats)]
        results = await asyncio.gather(*jobs)
        return EvalReport(self.name, list(results))

    async def _run_case(self, engine: Engine, case: Case, rep: int) -> CaseResult:
        res = CaseResult(case=case, outcome=None, repeat=rep)
        t0 = time.monotonic()
        try:
            inp = self.make_input(case) if self.make_input else case.input
            scope: Any = engine.tenant(self.tenant) if self.tenant else engine
            handle = await scope.start(self.algo, inp, budget=self.budget, meta={EVAL_META: True})
            res.run_id = handle.run_id
            outcome = await handle.result()
            res.outcome = outcome
            res.text = self.extract(outcome.value)
            res.latency_s = time.monotonic() - t0
            await fill_trajectory(engine, res)
        except Exception as e:
            res.error = f"{type(e).__name__}: {e}"
            res.latency_s = time.monotonic() - t0
            log.warning("eval case %s could not be run", case.id, exc_info=True)
        for m in self.metrics:
            res.scores.extend(await _score(m, res))
        return res


async def fill_trajectory(engine: Engine, res: CaseResult) -> None:
    """Fill `res` (which has a `run_id`) with the run's spans (whole trace, child runs included) and its spend."""
    assert res.run_id
    meta = await engine.project(RUNMETA, res.run_id)
    res.steps, res.tokens, res.cost = meta.ledger.steps, meta.ledger.tokens, meta.ledger.cost

    async def visit(run_id: str, parent: list[Any] | None, depth: int) -> None:
        evs = [await engine.hydrate(e) for e in await engine.store.read(run_id)]
        res.spans.extend(spans_from_events(evs, parent_events=parent, capture_content=True))
        if depth < 8:
            for child in children_of(evs):
                await visit(child, evs, depth + 1)

    await visit(res.run_id, None, 0)


async def _score(metric: Metric, res: CaseResult) -> list[Score]:
    if res.error is not None:
        return [
            Score(
                name=metric.name,
                value=False,
                reason=f"case not run: {res.error}"[:300],
                metadata={"error": True},
            )
        ]
    try:
        out = await metric.score(res)
    except Exception as e:
        log.warning("metric %s raised on %s", metric.name, res.case.id, exc_info=True)
        return [
            Score(
                name=metric.name,
                value=False,
                reason=f"metric error: {type(e).__name__}: {e}"[:300],
                metadata={"error": True},
            )
        ]
    return [out] if isinstance(out, Score) else list(out)
