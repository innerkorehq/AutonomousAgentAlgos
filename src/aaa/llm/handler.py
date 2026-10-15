from __future__ import annotations

import base64
import logging
from collections import OrderedDict
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from aaa.core.retry import RetryPolicy
from aaa.events import Event
from aaa.llm.client import LLMClient
from aaa.llm.errors import (
    ContextOverflow,
    FatalError,
    GuardrailBlocked,
    LLMError,
    StateNotFound,
    UnsupportedModality,
)
from aaa.llm.streaming import collect
from aaa.llm.types import LLMRequest, LLMResponse, Message
from aaa.runtime import Engine

if TYPE_CHECKING:
    from aaa.context import ContextPolicy, Prepared, SummaryBook
    from aaa.guard import GuardPolicy
    from aaa.guard.enforce import Enforcer

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
    the provider receives. Summaries it writes are persisted as `aaa.context.summary` events.

    With a server-state client (`ServerStateClient`, e.g. the OpenAI Responses API) the provider holds the
    conversation: a continuing request sends only the new messages plus the system prompt, referring to the
    provider's own handle (`state_id`, recorded on each reply). The full conversation is sent instead whenever
    the context policy had to rewrite it, the provider has lost the state, or no handle is known. Replies carry
    `state` = `continued` | `fresh` | `fallback` so the saving is visible in the log."""

    def __init__(
        self,
        client: LLMClient,
        *,
        concurrency: int = 8,
        emit_deltas: bool = True,
        context: ContextPolicy | None = None,
        guards: GuardPolicy | None = None,
    ) -> None:
        self.client, self.concurrency, self.emit_deltas = client, concurrency, emit_deltas
        self.context = context
        self.guards = guards
        self.enforcer: Enforcer | None = None
        self._books: OrderedDict[str, SummaryBook] = OrderedDict()
        self.engine: Engine | None = None
        # request event id -> the conversation AFTER that request's reply. A continuing request
        # (`conv.prev`) is resolved from here; a miss (restart, other process) rebuilds from the log.
        self._convs: OrderedDict[str, list[Message]] = OrderedDict()
        self._states: dict[
            str, str
        ] = {}  # request event id -> the provider's state handle after its reply
        self.max_cached_conversations = 1024
        self.rebuilds = 0

    def attach(self, engine: Engine) -> None:
        self.engine = engine
        if self.guards is not None:
            from aaa.guard.enforce import Enforcer

            self.enforcer = Enforcer(engine, self.guards)
        if self not in engine.state_holders:
            engine.state_holders.append(self)  # so erasing a run can delete what the provider holds
        engine.subscribe(REQUESTED, self.handle, concurrency=self.concurrency, group="aaa.llm")

    async def _run(self, e: Event, req: LLMRequest) -> LLMResponse:
        """One provider call, inside the guardrails: user input is checked (and possibly redacted) before it is sent,
        the reply before it is recorded."""
        enf = self.enforcer
        if enf is None:
            return await self._call(e, req)
        from aaa.guard.types import GuardContext

        if enf.policy.enforces("input"):
            msgs: list[Message] = []
            for m in req.messages:
                if m.role == "user" and (m.content or m.parts):
                    ctx = GuardContext(
                        "input", e.run_id, e.tenant, dict(e.meta), has_media=bool(m.parts)
                    )
                    dec = await enf.decide("input", m.content or "", ctx, e)
                    if dec.action in (
                        "block",
                        "approve",
                    ):  # no human is in this loop: an approval need is a stop
                        raise GuardrailBlocked("input", dec.categories, dec.blocked_by)
                    if dec.action == "redact" and dec.text is not None:
                        m = m.model_copy(update={"content": dec.text})
                msgs.append(m)
            req.messages = msgs
        resp = await self._call(e, req)
        if enf.policy.enforces("output") and resp.message.content:
            dec = await enf.decide(
                "output",
                resp.message.content,
                GuardContext("output", e.run_id, e.tenant, dict(e.meta)),
                e,
            )
            if dec.action in ("block", "approve"):
                raise GuardrailBlocked("output", dec.categories, dec.blocked_by)
            if dec.action == "redact" and dec.text is not None:
                resp.message.content = dec.text
        return resp

    async def _prepare_media(self, req: LLMRequest) -> LLMRequest:
        """Refuse attachments the client cannot send, and hand the client the bytes (blob references are resolved here, for this
        call only: the request on the log keeps the references)."""
        used = {p.type for m in req.messages for p in m.parts or []}
        if not used:
            return req
        supported = frozenset(getattr(self.client, "modalities", frozenset({"text"})))
        if missing := used - supported:
            raise UnsupportedModality(missing, supported)
        if self.engine is None:
            return req
        out: list[Message] = []
        for m in req.messages:
            if not m.parts:
                out.append(m)
                continue
            parts = []
            for p in m.parts:
                if p.blob is not None:
                    raw = await self.engine.media_bytes(p)
                    p = p.model_copy(update={"data": base64.b64encode(raw).decode(), "blob": None})
                parts.append(p)
            out.append(m.model_copy(update={"parts": parts}))
        return req.model_copy(update={"messages": out})

    async def _call(self, e: Event, req: LLMRequest) -> LLMResponse:
        req = await self._prepare_media(req)
        stream = getattr(self.client, "stream", None)
        # An enforcing output guard must see the whole reply before anyone does, so the reply is not streamed.
        buffered = self.enforcer is not None and self.enforcer.policy.enforces("output")
        if not (req.stream and stream) or buffered:
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
    def _remember(
        self,
        request_id: str,
        after_reply: list[Message],
        prev: str | None,
        state_id: str | None = None,
    ) -> None:
        if prev:  # the chain advanced: the older snapshot of it is superseded
            self._convs.pop(prev, None)
            self._states.pop(prev, None)
        self._convs[request_id] = after_reply
        if state_id:
            self._states[request_id] = state_id
        while len(self._convs) > self.max_cached_conversations:
            evicted, _ = self._convs.popitem(last=False)
            self._states.pop(evicted, None)

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
            if r.id == prev and reply.data.get("state_id"):
                self._states[prev] = reply.data["state_id"]
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

    # -- provider-held state ------------------------------------------------------------------
    @property
    def _stateful(self) -> bool:
        return bool(getattr(self.client, "server_state", False))

    def _continuation(
        self, e: Event, req: LLMRequest, transcript: list[Message]
    ) -> tuple[str, list[Message]] | None:
        """(state id, messages to send) if the provider can pick up where it left off, else None."""
        conv = e.data.get("conv")
        if conv:
            state_id = self._states.get(conv["prev"])
            fresh = transcript[len(transcript) - len(conv["delta"]) :]
        elif req.resume:
            state_id = req.resume.state_id
            seen, fresh = 0, []
            for m in transcript:
                if m.role == "system" and seen == 0:
                    continue  # leading system messages travel as instructions
                if seen >= req.resume.covered:
                    fresh.append(m)
                seen += m.role != "system"
        else:
            return None
        if not state_id or not fresh:
            return None
        leading: list[Message] = []
        for m in transcript:
            if m.role != "system":
                break
            leading.append(m)
        # tool results must name the PROVIDER's call ids, which differ from AAA's per-call ids
        pid = {c.id: c.provider_id for m in transcript for c in m.tool_calls if c.provider_id}
        fresh = [
            m.model_copy(update={"tool_call_id": pid.get(m.tool_call_id or "", m.tool_call_id)})
            if m.role == "tool"
            else m
            for m in fresh
        ]
        return state_id, [*leading, *fresh]

    async def forget_state(self, state_ids: Sequence[str]) -> int:
        """Delete provider-held state for these handles (used by `engine.erase`)."""
        fn = getattr(self.client, "forget_state", None)
        return int(await fn(list(state_ids))) if fn else 0

    async def handle(self, e: Event) -> list[Event]:
        tag = e.data.get("tag")
        preps: list[Prepared] = []
        mode = "fresh"
        try:
            req = LLMRequest.model_validate(e.data["request"])
            transcript = await self._transcript(e, req)
            req.messages, req.continue_from, resume, req.resume = transcript, None, req.resume, None
            if prep := await self._prepare(e, transcript, req):
                preps.append(prep)
                req.messages = prep.messages
            cont = None
            if self._stateful and (prep is None or prep.report is None):
                req.resume = resume
                cont = self._continuation(e, req, transcript)
                req.resume = None
            if cont:
                req.continue_from, req.messages = cont
                mode = "continued"
            try:
                resp = await self._run(e, req)
            except StateNotFound:
                # the provider no longer has our conversation: send all of it instead
                if not cont:
                    raise
                req.continue_from, req.messages = (
                    None,
                    (preps[-1].messages if preps else transcript),
                )
                mode = "fallback"
                resp = await self._run(e, req)
            except ContextOverflow:
                # our estimate was too optimistic: tighten the budget once and try again
                if not (retry := await self._prepare(e, transcript, req, pressure=0.5)):
                    raise
                preps.append(retry)
                req.continue_from, req.messages = None, retry.messages
                mode = "fresh"
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
        stateful = self._stateful
        for i, call in enumerate(resp.message.tool_calls):
            if stateful and call.provider_id is None:
                call.provider_id = call.id  # the provider matches results by ITS id: keep it
            call.id = f"call_{e.id}_{i}"
        # the LOGICAL conversation (not the compacted view) is what later deltas append to
        self._remember(
            e.id,
            [*transcript, resp.message],
            (e.data.get("conv") or {}).get("prev"),
            resp.state_id if stateful else None,
        )
        report: dict[str, Any] | None = None
        if preps and self.context is not None:
            report = preps[-1].report
            if (
                not req.continue_from
            ):  # a continued request's prompt count covers the whole server-side chat
                self.context.observe(req.model, req.messages, req.tools, resp.usage.prompt_tokens)
        dumped = resp.model_dump(mode="json")
        if dumped.get("state_id") is None:  # new optional fields stay out of ordinary events
            dumped.pop("state_id", None)
        for c in dumped["message"]["tool_calls"]:
            if c.get("provider_id") is None:
                c.pop("provider_id", None)
        return [
            *self._summary_events(e, preps),
            _result(
                e,
                COMPLETED,
                {
                    "tag": tag,
                    "response": dumped,
                    "usage": resp.usage.model_dump(),
                    "cost": resp.cost,
                    **({"context": report} if report else {}),
                    **(
                        {"state_id": resp.state_id, "state": mode}
                        if stateful and resp.state_id
                        else {}
                    ),
                },
            ),
        ]
