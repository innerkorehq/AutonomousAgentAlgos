"""Guardrails inside the engine: stored verdicts, recording, and the decision handlers act on.

Identity of a verdict is `(run, guard, guard version, stage, content hash)`. The first time a run shows a guard some content, the
guard is called and its verdict is appended to the log (`aaa.guard.verdict`, with the content's hash, never the content). Any later
request that carries the same content, including a redelivery after a crash, finds that verdict and does not call the guard
again: a classifier is billable and not always deterministic, so the stored verdict wins. Two exceptions:

* a **redact** verdict has no stored text (the log never holds the content), so redaction is recomputed; guards that redact
  (regex, Presidio) are deterministic. (That recomputation is also why redaction needs no log lookup to stay consistent.)
* a verdict produced because the **guard itself failed** (timeout, outage) is recorded for audit but never reused, so a
  transient outage does not become a permanent block."""

from __future__ import annotations

import logging
from collections import OrderedDict

from aaa.events import Event
from aaa.guard.policy import Decision, GuardPolicy, GuardSpec, Verdict, content_hash
from aaa.guard.types import Finding, GuardContext, GuardResult, Stage
from aaa.runtime import Engine

log = logging.getLogger("aaa.guard")
VERDICT = "aaa.guard.verdict"


def verdict_id(stream: str, spec: GuardSpec, stage: Stage, h: str) -> str:
    return f"{stream}:guard:{spec.name}:{spec.guard.version}:{stage}:{h}"


class Enforcer:
    def __init__(self, engine: Engine, policy: GuardPolicy, *, cache_size: int = 4096) -> None:
        self.engine, self.policy = engine, policy
        self._seen: OrderedDict[str, Verdict] = OrderedDict()
        self.cache_size = cache_size
        self.calls = 0  # guard invocations (not counting reused verdicts); for tests and metrics

    def _remember(self, vid: str, v: Verdict) -> None:
        self._seen[vid] = v
        self._seen.move_to_end(vid)
        while len(self._seen) > self.cache_size:
            self._seen.popitem(last=False)

    async def _stored(self, request: Event, vid: str) -> Verdict | None:
        eng = self.engine
        if not await eng.retry(lambda: eng.store.has(request.stream, vid), "read"):
            return None
        for e in await eng.retry(lambda: eng.store.read(request.stream), "read"):
            if e.id == vid:
                d = e.data
                res = GuardResult(
                    d["action"],
                    [Finding(c, (d.get("scores") or {}).get(c)) for c in d.get("categories") or []],
                )
                return Verdict(
                    d["guard"],
                    d["version"],
                    d["stage"],
                    d["mode"],
                    res,
                    reused=True,
                    content_hash=d.get("hash", ""),
                )
        return None

    async def decide(
        self,
        stage: Stage,
        content: str,
        ctx: GuardContext,
        request: Event,
        *,
        mode: str = "enforce",
    ) -> Decision:
        async def runner(spec: GuardSpec, text: str, c: GuardContext) -> Verdict:
            h = content_hash(text + ("\0media" if c.has_media else ""))
            vid = verdict_id(request.stream, spec, c.stage, h)
            cached = self._seen.get(vid) or await self._stored(request, vid)
            if cached is not None and cached.result.action != "redact":
                self._remember(vid, cached)
                return cached
            self.calls += 1
            v = await self.policy.run_one(spec, text, c)
            data = v.data()
            if c.tool:
                data["tool"] = c.tool
            await self.engine.emit(
                request.caused(VERDICT, data, id=vid + (":error" if v.error else ""))
            )
            if not v.error:
                self._remember(vid, v)
            return v

        return await self.policy.evaluate(stage, content, ctx, runner=runner, mode=mode)
