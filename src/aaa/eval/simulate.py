"""Multi-turn evaluation: a model plays the user, a real `Session` plays the agent.

    user = SimulatedUser(client, persona="a impatient customer", goal="get a refund for order A1")
    runner = SessionEvalRunner(manager, user, [Contains("refund")], max_turns=6)
    report = await runner.run(engine, Dataset.from_list([{"input": {"opening": "Hi, where is my refund?"}}]))

Each case is one conversation, bounded three ways: `max_turns`, the user model's own stop marker, and a `max_cost` for the
user model. The agent side is an ordinary session (so its context, memory and tools behave exactly as in production) stamped
`aaa_eval` so online evaluators ignore it. Metrics see the whole transcript as `result.text`, and the pairs as `result.turns`.
A case's `input` may carry its own `persona`, `goal` and `opening`, which override the simulator's defaults."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Sequence
from typing import Any

from aaa.core.outcome import Outcome, OutcomeKind
from aaa.eval.dataset import Case, Dataset
from aaa.eval.metrics import Metric
from aaa.eval.report import EvalReport
from aaa.eval.result import CaseResult
from aaa.eval.suite import EVAL_META, _score, fill_trajectory
from aaa.llm.types import LLMRequest, Message
from aaa.runtime import Engine
from aaa.session import SessionManager

log = logging.getLogger("aaa.eval")
DONE = "[DONE]"
_SYSTEM = (
    "You are role-playing a person talking to a company's AI assistant, to test it. Stay in character and write ONLY "
    "the next message the person would send: short, natural, in their voice. Never reveal that you are simulating.\n"
    "Persona: {persona}\nGoal: {goal}\n"
    f"When the goal is achieved, or the assistant has clearly failed or refused for good, reply with exactly {DONE}."
)


class SimulatedUser:
    """An LLM-driven user. `client` is any `LLMClient`; its calls are counted in `spent`/`calls` and capped by `max_cost`."""

    def __init__(
        self,
        client: Any,
        *,
        persona: str = "a polite customer",
        goal: str = "get the question answered",
        opening: str | None = None,
        model: str | None = None,
        max_cost: float | None = None,
    ) -> None:
        self.client, self.persona, self.goal, self.opening = client, persona, goal, opening
        self.model, self.max_cost = model, max_cost
        self.spent, self.calls = 0.0, 0
        self._root: SimulatedUser | None = (
            None  # the simulator a per-case copy reports its spend to
        )

    def for_case(self, case: Case) -> SimulatedUser:
        """A copy using the case's own persona/goal/opening where it gives them (the cost counters stay shared)."""
        inp = case.input
        twin = SimulatedUser(
            self.client,
            persona=str(inp.get("persona", self.persona)),
            goal=str(inp.get("goal", self.goal)),
            opening=inp.get("opening", self.opening),
            model=self.model,
            max_cost=self.max_cost,
        )
        twin._root = self._root or self
        return twin

    def _charge(self, cost: float) -> None:
        for u in (self, self._root):
            if u is not None:
                u.spent += cost
                u.calls += 1

    async def next_message(self, transcript: Sequence[tuple[str, str]]) -> str | None:
        """The user's next message given the (user, assistant) pairs so far, or `None` when the conversation is over."""
        if not transcript and self.opening:
            return self.opening
        root = self._root or self
        if self.max_cost is not None and root.spent >= self.max_cost:
            return None
        msgs = [
            Message(role="system", content=_SYSTEM.format(persona=self.persona, goal=self.goal))
        ]
        for u, a in transcript:
            msgs += [Message(role="assistant", content=u), Message(role="user", content=a)]
        if not transcript:
            msgs.append(
                Message(
                    role="user",
                    content="(The conversation has not started. Write your first message.)",
                )
            )
        resp = await self.client.complete(LLMRequest(model=self.model, messages=msgs))
        self._charge(float(getattr(resp, "cost", 0.0) or 0.0))
        text = (resp.message.content or "").strip()
        return None if not text or text.startswith(DONE) else text


class SessionEvalRunner:
    """Run each case of a dataset as a conversation between `user` and a session of `manager`."""

    def __init__(
        self,
        manager: SessionManager,
        user: SimulatedUser,
        metrics: Sequence[Metric],
        *,
        max_turns: int = 6,
        concurrency: int = 2,
        turn_timeout_s: float = 120.0,
        tenant: str | None = None,
        name: str = "conversations",
    ) -> None:
        if not 1 <= max_turns <= 30:
            raise ValueError("max_turns must be between 1 and 30")
        self.manager, self.user, self.metrics = manager, user, list(metrics)
        self.max_turns, self.concurrency, self.turn_timeout_s = (
            max_turns,
            max(1, concurrency),
            turn_timeout_s,
        )
        self.tenant, self.name = tenant, name

    async def run(self, engine: Engine, dataset: Dataset) -> EvalReport:
        sem = asyncio.Semaphore(self.concurrency)

        async def one(case: Case) -> CaseResult:
            async with sem:
                return await self._run_case(engine, case)

        return EvalReport(self.name, list(await asyncio.gather(*(one(c) for c in dataset))))

    async def _run_case(self, engine: Engine, case: Case) -> CaseResult:
        res = CaseResult(case=case, outcome=None)
        t0 = time.monotonic()
        user = self.user.for_case(case)
        try:
            session = await self.manager.create(
                tenant=self.tenant,
                meta={EVAL_META: True},
            )
            res.run_id = session.id
            last_kind, last_reason = OutcomeKind.SUCCESS.value, None
            try:
                for _ in range(self.max_turns):
                    msg = await user.next_message(res.turns)
                    if msg is None:
                        break
                    turn = await session.send(msg, timeout_s=self.turn_timeout_s)
                    res.turns.append((msg, turn.text))
                    if turn.run_id:
                        part = CaseResult(case=case, outcome=None, run_id=turn.run_id)
                        await fill_trajectory(engine, part)
                        res.spans += part.spans
                        res.steps, res.tokens, res.cost = (
                            res.steps + part.steps,
                            res.tokens + part.tokens,
                            res.cost + part.cost,
                        )
                    if not turn.ok:
                        last_kind, last_reason = turn.kind, turn.reason
                        break
            finally:
                await session.close("eval finished")
            res.text = "\n".join(f"User: {u}\nAssistant: {a}" for u, a in res.turns)
            res.outcome = Outcome(
                kind=OutcomeKind(last_kind)
                if last_kind in OutcomeKind._value2member_map_
                else OutcomeKind.FAILED,
                value={
                    "turns": len(res.turns),
                    "last_reply": res.turns[-1][1] if res.turns else "",
                },
                reason=last_reason,
            )
        except Exception as e:
            res.error = f"{type(e).__name__}: {e}"
            log.warning("conversation %s could not be run", case.id, exc_info=True)
        res.latency_s = time.monotonic() - t0
        for m in self.metrics:
            res.scores.extend(await _score(m, res))
        return res
