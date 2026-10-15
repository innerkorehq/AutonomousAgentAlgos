"""The context policy: how big a request may be, and what to do when it would be bigger.

The policy is applied by `LLMHandler(client, context=policy)` to every request, at the last moment before it is sent.
The conversation the *algorithm* sees (and the event log records) is never rewritten: compaction is a view
over it. That keeps the delta-request scheme (`conv: {prev, delta}`) and replay intact, while the provider
receives a conversation that fits.

When does it act? Compaction starts when the estimate passes `trigger` of the input budget and stops once it is
under `target`, so it runs occasionally (and a summary, once made, is reused until the conversation outgrows
it again) instead of on every turn. Estimates are calibrated against the provider's own token counts."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from aaa.context.strategies import (
    ClearToolResults,
    DropOldest,
    Strategy,
    Summarize,
    SummaryRecord,
    Work,
    head_end,
    prefix_hash,
    summary_message,
)
from aaa.context.tokens import CalibratedCounter, HeuristicCounter, TokenCounter
from aaa.llm.client import LLMClient
from aaa.llm.types import Message, ToolSpec

from aaa.context.strategies import ClearMedia, TruncateToolResults  # isort: skip


@dataclass
class SummaryBook:
    """Summaries known for one conversation (loaded from the log, or made in this process)."""

    records: list[SummaryRecord] = field(default_factory=list)

    def add(self, rec: SummaryRecord) -> None:
        if not any(r.upto == rec.upto and r.prefix_hash == rec.prefix_hash for r in self.records):
            self.records.append(rec)

    def best(self, transcript: Sequence[Message]) -> SummaryRecord | None:
        """The summary covering the longest prefix of `transcript` that matches by content."""
        for rec in sorted(self.records, key=lambda r: r.upto, reverse=True):
            if (
                0 < rec.upto <= len(transcript)
                and prefix_hash(transcript[: rec.upto]) == rec.prefix_hash
            ):
                return rec
        return None


@dataclass
class Prepared:
    messages: list[Message]
    report: dict[str, Any] | None = None  # None: left untouched
    new_summaries: list[SummaryRecord] = field(default_factory=list)
    tokens_spent: int = 0
    cost: float = 0.0


class ContextPolicy:
    """Fit conversations into a model's input window.

    `max_input_tokens` is the model's context window (an int, or a function of the model name). Nothing here knows
    your model's real limit, so say it: a too-large number means overflow errors, a too-small one wastes context.
    `reserve_output` is kept free for the reply. `strategies` run in order until the conversation is under
    `target`; the default is cheap-and-lossless first, `Summarize` only if you give a `summarizer` client."""

    def __init__(
        self,
        max_input_tokens: int | Callable[[str | None], int],
        *,
        reserve_output: int = 4_096,
        trigger: float = 0.85,
        target: float = 0.60,
        keep_recent: int = 3,
        strategies: Sequence[Strategy] | None = None,
        counter: TokenCounter | None = None,
        summarizer: LLMClient | None = None,
        summarizer_model: str | None = None,
    ) -> None:
        if not 0 < target < trigger <= 1:
            raise ValueError("need 0 < target < trigger <= 1")
        self.window = (
            max_input_tokens if callable(max_input_tokens) else (lambda _m, _n=max_input_tokens: _n)
        )
        self.reserve_output, self.trigger, self.target, self.keep_recent = (
            reserve_output,
            trigger,
            target,
            keep_recent,
        )
        self.counter: TokenCounter = counter or CalibratedCounter(HeuristicCounter())
        if strategies is None:
            default: list[Strategy] = [TruncateToolResults(), ClearMedia(), ClearToolResults()]
            if summarizer is not None:
                default.append(Summarize(summarizer, summarizer_model))
            default.append(DropOldest())
            strategies = default
        self.strategies: list[Strategy] = list(strategies)
        self.summarizes = any(isinstance(s, Summarize) for s in self.strategies)

    # -- budgeting ---------------------------------------------------------------------------
    def budget(self, model: str | None) -> int:
        return max(1, self.window(model) - self.reserve_output)

    def estimate(
        self, messages: Sequence[Message], tools: Sequence[ToolSpec] = (), model: str | None = None
    ) -> int:
        if isinstance(self.counter, CalibratedCounter):
            self.counter.model = model
        return self.counter.count(messages, tools)

    def observe(
        self,
        model: str | None,
        sent: Sequence[Message],
        tools: Sequence[ToolSpec],
        actual_prompt_tokens: int,
    ) -> None:
        """Calibrate on a real response: compare the provider's own prompt-token count with our raw estimate."""
        if isinstance(self.counter, CalibratedCounter):
            self.counter.observe(model, self.counter.base.count(sent, tools), actual_prompt_tokens)

    # -- the work ----------------------------------------------------------------------------
    async def prepare(
        self,
        transcript: Sequence[Message],
        *,
        tools: Sequence[ToolSpec] = (),
        model: str | None = None,
        book: SummaryBook | None = None,
        pressure: float = 1.0,
    ) -> Prepared:
        """The messages to send for `transcript`. `pressure < 1` assumes the real size is `1/pressure` times our
        estimate, which forces deeper cuts. It is used to retry after the provider itself reported an overflow
        that our estimate missed: strategies stop as soon as the *estimate* fits, so re-running them unchanged
        would just produce the same request."""
        budget = self.budget(model)
        inflate = 1.0 / pressure
        before = self.estimate(transcript, tools, model)
        rec = book.best(transcript) if book else None
        view = list(transcript)
        if (
            rec is not None
        ):  # a stored summary: head + summary + the rest (a stable, cache-friendly prefix)
            h = head_end(view)
            view = [*view[:h], summary_message(rec.text), *view[rec.upto :]]
        now = self.estimate(view, tools, model)
        if now * inflate <= int(budget * self.trigger) and (rec is not None or pressure >= 1.0):
            report = None if rec is None else self._report(before, now, ["stored_summary"])
            return Prepared(view, report)
        work = Work(
            messages=view,
            tools=tools,
            counter=_Bound(self, model, inflate),
            target=int(budget * self.target),
            keep_recent=self.keep_recent,
            summary_upto=rec.upto if rec else 0,
            original=transcript,
            applied=["stored_summary"] if rec else [],
        )
        for strategy in self.strategies:
            if work.fits():
                break
            await strategy.apply(work)
        after = round(work.tokens() / inflate)
        return Prepared(
            work.messages,
            self._report(before, after, work.applied, over=work.tokens() > work.target),
            work.new_summaries,
            work.tokens_spent,
            work.cost,
        )

    @staticmethod
    def _report(before: int, after: int, applied: list[str], over: bool = False) -> dict[str, Any]:
        return {
            "tokens_before": before,
            "tokens_after": after,
            "applied": applied,
            **({"still_over_target": True} if over else {}),
        }


class _Bound:
    """Counter adapter handing the policy's model-aware (and, on a retry, inflated) estimate to strategies."""

    def __init__(self, policy: ContextPolicy, model: str | None, inflate: float = 1.0) -> None:
        self.policy, self.model, self.inflate = policy, model, inflate

    def count(self, messages: Sequence[Message], tools: Sequence[ToolSpec] = ()) -> int:
        return math.ceil(self.policy.estimate(messages, tools, self.model) * self.inflate)
