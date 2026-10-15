"""Turn real production runs into eval cases: `await Dataset.from_traces(engine, ...)`.

Production traffic is the best source of test cases and the most dangerous one: inputs hold customer data. So the default is to
scrub every input through the standard redactor before it becomes a case, runs whose payloads were erased are skipped, and
nothing about the tenant is copied unless you ask."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

from aaa.core.outcome import OutcomeKind
from aaa.eval.dataset import Case, Dataset
from aaa.eval.result import default_extract
from aaa.events import Event, RunInfo
from aaa.events.event import ERASED
from aaa.privacy import Redactor, standard_redactor
from aaa.runtime import Engine

EVAL_META = "aaa_eval"


async def dataset_from_traces(
    engine: Engine,
    *,
    filter: Callable[[RunInfo], bool] | None = None,
    algo: str | None = None,
    tenant: str | None = None,
    outcomes: tuple[str, ...] | None = (OutcomeKind.SUCCESS.value,),
    since: datetime | None = None,
    limit: int = 100,
    scrub: Redactor | bool = True,
    expected: str | None = None,
    include_tenant: bool = False,
    name: str = "traces",
) -> Dataset:
    """Build a `Dataset` from finished top-level runs in `engine`'s store.

    * Skipped always: child runs, runs started by an eval (`aaa_eval`), runs still open, runs whose payloads were erased, session
      runs (their turns are runs of the inner algorithm; build session cases by hand).
    * `algo` / `tenant` / `outcomes` / `since` / `filter` narrow the selection; the newest `limit` matching runs are kept.
    * `scrub=True` redacts each input with `standard_redactor()` (emails, cards, tokens ...); pass your own `Redactor`, or `False` to
      keep inputs verbatim (only for data you are allowed to copy).
    * `expected="output"` records what the run answered as the case's expected value: a golden file for regression testing, which
      pins down CURRENT behaviour, not correct behaviour. Review it before trusting it.
    * The case id is the run id (so a case can be traced back); `include_tenant` also copies the tenant into the metadata."""
    redactor: Redactor | None = (
        standard_redactor() if scrub is True else (scrub if scrub is not False else None)
    )
    rows = []
    for info in await engine.retry(engine.store.run_index, "read"):
        s, c = info.started, info.completed
        if c is None or s.data.get("parent") or s.meta.get(EVAL_META) or s.data.get(ERASED):
            continue
        if str(s.data.get("algo", "")).startswith("aaa.session:"):
            continue
        if algo is not None and s.data.get("algo") != algo:
            continue
        if tenant is not None and s.tenant != tenant:
            continue
        if outcomes is not None and c.data.get("kind") not in outcomes:
            continue
        if since is not None and c.time < since:
            continue
        if filter is not None and not filter(info):
            continue
        rows.append(info)
    rows.sort(key=lambda r: (r.completed.time if r.completed else r.started.time, r.stream))
    cases = []
    for info in rows[-limit:]:
        started: Event = await engine.hydrate(info.started)
        if started.data.get("input") is None:
            continue
        if redactor is not None:
            started = redactor.redact(started)
        inp = dict(started.data.get("input") or {})
        completed = info.completed
        assert completed is not None
        cases.append(
            Case(
                id=info.stream,
                input=inp,
                expected=(
                    default_extract((await engine.hydrate(completed)).data.get("value"))
                    if expected == "output"
                    else None
                ),
                tags=[
                    t
                    for t in (str(started.data.get("algo", "")), str(completed.data.get("kind")))
                    if t
                ],
                metadata={
                    "source_run": info.stream,
                    **({"tenant": info.started.tenant} if include_tenant else {}),
                },
            )
        )
    return Dataset(cases, name)
