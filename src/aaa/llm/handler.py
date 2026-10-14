from __future__ import annotations

import logging
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

from aaa.core.retry import RetryPolicy
from aaa.events import Event
from aaa.llm.client import LLMClient
from aaa.llm.errors import ContextOverflow, FatalError, LLMError
from aaa.llm.streaming import collect
from aaa.llm.types import LLMRequest, LLMResponse, Message
from aaa.runtime import Engine

if TYPE_CHECKING:
    from aaa.context import ContextPolicy, Prepared, SummaryBook

log = logging.getLogger("aaa.llm")

REQUESTED = "aaa.llm.requested"
COMPLETED = "aaa.llm.completed"
FAILED = "aaa.llm.failed"
DELTA = "aaa.llm.delta"
CONTEXT_SUMMARY = "aaa.context.summary"


def _result(request: Event, type: str, data: dict) -> Event:
    """One result per request on the log, whichever worker finishes first (deterministic id)."""
    return request.caused(type, data, id=f"{request.id}:result")


class LLMHandler:
    """Capability handler: `aaa.llm.requested` in -> `aaa.llm.completed|failed` out.
    Request data: {"request": LLMRequest, "tag": optional str echoed back}. Failures are facts
    on the log (the algorithm decides what to do), never exceptions. The completed event
    carries top-level `usage`/`cost` so the Engine's budget ledger sees them. With
    `request.stream` and a streaming client, `aaa.llm.delta` events are emitted as text arrives.
    Tool-call ids are rewritten to be unique per call (providers may omit or repeat them).

    With `context=` (an `aaa.context.ContextPolicy`) each request is fitted to the model's window just
    before it is sent: the conversation the algorithm and the log hold is never rewritten, only the copy
    the provider receives. Summaries it writes are persisted as `aaa.context.summary` events."""

    def __init__(
        self,
        client: LLMClient,
        *,
        concurrency: int = 8,
        emit_deltas: bool = True,
        context: ContextPolicy | None = None,
    ) -> None:
        self.client, self.concurrency, self.emit_deltas = client, concurrency, emit_deltas
        self.context = context
        self._books: OrderedDict[str, SummaryBook] = OrderedDict()
        self.engine: Engine | None = None
        # request event id -> the conversation AFTER that request's reply. A continuing request
        # (`conv.prev`) is resolved from here; a miss (restart, other process) rebuilds from the log.
        self._convs: OrderedDict[str, list[Message]] = OrderedDict()
        self.max_cached_conversations = 1024
        self.rebuilds = 0

    def attach(self, engine: Engine) -> None:
        self.engine = engine
        engine.subscribe(REQUESTED, self.handle, concurrency=self.concurrency, group="aaa.llm")

    async def _run(self, e: Event, req: LLMRequest) -> LLMResponse:
        stream = getattr(self.client, "stream", None)
        if not (req.stream and stream):
            return await self.client.complete(req)

        async def gen():
            n = 0
            async for ch in stream(req):
                if ch.text and self.emit_deltas and self.engine:
                    n += 1
                    try:  # deltas are a live view: losing one must never fail (and re-bill) the call
                        await self.engine.emit(
                            e.caused(
                                DELTA,
                                {"text": ch.text, "tag": e.data.get("tag")},
                                id=f"{e.id}:d{n}",
                            ),
                            retry=RetryPolicy(max_attempts=1),
                        )
                    except Exception:
                        log.warning("could not publish a stream delta; continuing", exc_info=True)
                yield ch

        return await collect(gen())

    # -- conversation reconstruction -------------------------------------------------------
    def _remember(self, request_id: str, after_reply: list[Message], prev: str | None) -> None:
        if prev:  # the chain advanced: the older snapshot of it is superseded
            self._convs.pop(prev, None)
        self._convs[request_id] = after_reply
        while len(self._convs) > self.max_cached_conversations:
            self._convs.popitem(last=False)

    async def _rebuild(self, stream: str, prev: str) -> list[Message]:
        """Reconstruct the conversation after `prev`'s reply from the run's log (cache miss)."""
        assert self.engine
        self.rebuilds += 1
        requests: dict[str, Event] = {}
        replies: dict[str, Event] = {}
        eng = self.engine
        for ev in await eng.retry(lambda: eng.store.read(stream), "read"):
            if ev.type == REQUESTED:
                requests[ev.id] = ev
            elif ev.type == COMPLETED and ev.id.endswith(":result"):
                replies[ev.id[: -len(":result")]] = ev
        chain: list[Event] = []
        cur: str | None = prev
        while cur is not None:
            if cur not in requests or cur not in replies:
                raise FatalError(
                    f"cannot rebuild conversation: request {cur} or its reply is not in the log"
                )
            chain.append(await self.engine.hydrate(requests[cur]))
            cur = (chain[-1].data.get("conv") or {}).get("prev")
        msgs: list[Message] = []
        for r in reversed(chain):
            conv = r.data.get("conv")
            msgs = (
                [*msgs, *(Message.model_validate(m) for m in conv["delta"])]
                if conv
                else [Message.model_validate(m) for m in r.data["request"]["messages"]]
            )
            reply = await self.engine.hydrate(replies[r.id])
            msgs.append(LLMResponse.model_validate(reply.data["response"]).message)
        return msgs

    async def _transcript(self, e: Event, req: LLMRequest) -> list[Message]:
        conv = e.data.get("conv")
        if not conv:
            return req.messages
        base = self._convs.get(conv["prev"])
        if base is None:
            base = await self._rebuild(e.stream, conv["prev"])
        return [*base, *(Message.model_validate(m) for m in conv["delta"])]

    # -- context management ----------------------------------------------------------------
    async def _book(self, stream: str) -> SummaryBook:
        """Summaries already written for this run (from the log, once per process per run)."""
        from aaa.context import SummaryBook, SummaryRecord

        if (book := self._books.get(stream)) is not None:
            self._books.move_to_end(stream)
            return book
        assert self.engine
        eng = self.engine
        book = SummaryBook()
        for ev in await eng.retry(lambda: eng.store.read(stream), "read"):
            if ev.type == CONTEXT_SUMMARY:
                d = ev.data
                book.add(
                    SummaryRecord(upto=d["upto"], prefix_hash=d["prefix_hash"], text=d["text"])
                )
        self._books[stream] = book
        while len(self._books) > 256:
            self._books.popitem(last=False)
        return book

    async def _prepare(
        self, e: Event, transcript: list[Message], req: LLMRequest, pressure: float = 1.0
    ) -> Prepared | None:
        if self.context is None:
            return None
        book = await self._book(e.stream) if self.context.summarizes else None
        prep = await self.context.prepare(
            transcript, tools=req.tools, model=req.model, book=book, pressure=pressure
        )
        for rec in prep.new_summaries:
            if book is not None:
                book.add(rec)
        return prep

    @staticmethod
    def _summary_events(e: Event, preps: list[Prepared]) -> list[Event]:
        out: list[Event] = []
        for p in preps:
            for n, rec in enumerate(p.new_summaries):
                out.append(
                    e.caused(
                        CONTEXT_SUMMARY,
                        {
                            "upto": rec.upto,
                            "prefix_hash": rec.prefix_hash,
                            "text": rec.text,
                            # what writing the summary cost, so the run's budget ledger sees it
                            **(
                                {"usage": {"total_tokens": p.tokens_spent}, "cost": p.cost}
                                if n == 0
                                else {}
                            ),
                        },
                        id=f"{e.id}:summary:{rec.upto}",
                    )
                )
        return out

    async def handle(self, e: Event) -> list[Event]:
        tag = e.data.get("tag")
        preps: list[Prepared] = []
        try:
            req = LLMRequest.model_validate(e.data["request"])
            transcript = await self._transcript(e, req)
            req.messages = transcript
            if prep := await self._prepare(e, transcript, req):
                preps.append(prep)
                req.messages = prep.messages
            try:
                resp = await self._run(e, req)
            except ContextOverflow:
                # our estimate was too optimistic: tighten the budget once and try again
                if not (retry := await self._prepare(e, transcript, req, pressure=0.5)):
                    raise
                preps.append(retry)
                req.messages = retry.messages
                resp = await self._run(e, req)
        except LLMError as exc:
            return [
                *self._summary_events(e, preps),
                _result(
                    e,
                    FAILED,
                    {
                        "tag": tag,
                        "error": {
                            "type": type(exc).__name__,
                            "message": str(exc),
                            "retryable": exc.retryable,
                        },
                    },
                ),
            ]
        for i, call in enumerate(resp.message.tool_calls):
            call.id = f"call_{e.id}_{i}"
        # the LOGICAL conversation (not the compacted view) is what later deltas append to
        self._remember(e.id, [*transcript, resp.message], (e.data.get("conv") or {}).get("prev"))
        report: dict[str, Any] | None = None
        if preps and self.context is not None:
            report = preps[-1].report
            self.context.observe(req.model, req.messages, req.tools, resp.usage.prompt_tokens)
        return [
            *self._summary_events(e, preps),
            _result(
                e,
                COMPLETED,
                {
                    "tag": tag,
                    "response": resp.model_dump(mode="json"),
                    "usage": resp.usage.model_dump(),
                    "cost": resp.cost,
                    **({"context": report} if report else {}),
                },
            ),
        ]
