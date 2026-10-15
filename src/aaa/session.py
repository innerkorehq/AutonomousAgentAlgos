"""Chat sessions: a durable conversation thread that outlives any single run.

    sessions = SessionManager(engine, ToolLoop(registry, stream=True), system="You are a helpful analyst.")
    chat = await sessions.create(user="ann")                  # or engine.tenant("acme") -> sessions.tenant("acme")
    reply = await chat.send("What were Q3 sales?")            # TurnResult(text=..., ok=True, ...)
    reply = await chat.send("And Q4?")                        # remembers the first question
    async with chat.stream("Summarize") as s:                 # token-by-token
        async for piece in s: print(piece, end="")
    await chat.close()

How it is built (nothing here is a new subsystem; it is the engine's own parts):

* A session is a **long-lived run** whose stream records `aaa.session.message` (the user speaking) and
  `aaa.session.reply` (the agent answering). So it gets tenancy, a whole-conversation budget, crash recovery,
  erasure and retention for free: close it and it is a finished run like any other.
* Each user message starts a **child run** of the inner algorithm with the conversation so far as its input.
  The `SessionAlgo` records the reply when the child finishes, in whichever process is handling the event, so a
  turn survives the death of the process that sent it: `await chat.wait(n)` from anywhere returns the reply.
* One turn at a time: a second `send` while a turn is running raises `SessionBusy`.
* Long conversations: pass `context=ContextPolicy(...)` and old turns are summarized once, persisted as
  `aaa.session.summary` events, and reused by content hash (the same machinery as run-level context management).
* Provider-held state: when the LLM client keeps conversations (`ServerStateClient`), each turn tells the provider to
  continue from the previous turn's state instead of resending the whole history."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Self

from pydantic import BaseModel, Field

from aaa.core.aio import wait_for
from aaa.core.errors import AAAError
from aaa.core.ids import new_id
from aaa.core.outcome import OutcomeKind
from aaa.events import Event, Projection
from aaa.llm.types import Message
from aaa.runtime.budget import Budget
from aaa.runtime.engine import RUN_COMPLETED, RUNMETA, Algorithm, Ctx, Engine

if TYPE_CHECKING:
    from aaa.context import ContextPolicy

log = logging.getLogger("aaa.session")

MESSAGE = "aaa.session.message"
REPLY = "aaa.session.reply"
SUMMARY = "aaa.session.summary"
CHILD_COMPLETED = "aaa.child.completed"


class SessionBusy(AAAError):
    """A turn is still running; one turn at a time per session."""


class SessionClosed(AAAError):
    """The session has ended (closed, or its budget ran out)."""


# ---- state: a fold of the session stream ------------------------------------------------------
class Turn(BaseModel):
    n: int
    user: str
    assistant: str | None = None
    status: Literal["pending", "done", "failed"] = "pending"
    run_id: str | None = None  # the child run that produced the reply
    state_id: str | None = None  # the provider's handle for the conversation after this turn
    reason: str | None = None


class SummaryEntry(BaseModel):
    upto: int
    prefix_hash: str
    text: str


class SessionState(BaseModel):
    turns: list[Turn] = Field(default_factory=list)
    summaries: list[SummaryEntry] = Field(default_factory=list)


def _apply(st: SessionState, e: Event) -> SessionState:
    d = e.data
    if e.type == MESSAGE:
        if d["turn"] == len(st.turns):  # a duplicate or out-of-order message is ignored
            st.turns.append(Turn(n=d["turn"], user=d["text"]))
    elif e.type == REPLY:
        if 0 <= d["turn"] < len(st.turns):
            t = st.turns[d["turn"]]
            ok = d.get("kind") == OutcomeKind.SUCCESS.value
            t.assistant, t.status = (d.get("text") or "", "done" if ok else "failed")
            t.run_id, t.state_id, t.reason = d.get("run_id"), d.get("state_id"), d.get("reason")
    elif e.type == SUMMARY:
        st.summaries.append(
            SummaryEntry(upto=d["upto"], prefix_hash=d["prefix_hash"], text=d["text"])
        )
    return st


SESSION = Projection(SessionState, _apply, name="aaa.session", version=1, state_type=SessionState)


def history_messages(st: SessionState, system: str = "") -> list[Message]:
    """The conversation so far as model messages: finished turns only (a failed turn leaves no trace)."""
    msgs = [Message(role="system", content=system)] if system else []
    for t in st.turns:
        if t.status == "done":
            msgs += [
                Message(role="user", content=t.user),
                Message(role="assistant", content=t.assistant or ""),
            ]
    return msgs


# ---- the algorithm that drives a session --------------------------------------------------------
class SessionAlgo:
    """Reacts to `aaa.session.message` by starting the inner algorithm on the conversation, and to the child's
    completion by recording `aaa.session.reply`. Stateless, like every algorithm: all state is the stream."""

    def __init__(
        self,
        inner: Algorithm,
        *,
        system: str = "",
        context: ContextPolicy | None = None,
        turn_budget: Budget | None = None,
    ) -> None:
        self.inner, self.system, self.context, self.turn_budget = (
            inner,
            system,
            context,
            turn_budget,
        )
        self.name = f"aaa.session:{inner.name}"

    def subscriptions(self) -> list[str]:
        return [MESSAGE, CHILD_COMPLETED]

    async def on_event(self, event: Event, ctx: Ctx, /) -> list[Event]:
        if event.type == MESSAGE:
            return await self._start_turn(event, ctx)
        if event.type == CHILD_COMPLETED:
            return await self._record_reply(event, ctx)
        return []

    async def _start_turn(self, e: Event, ctx: Ctx) -> list[Event]:
        st = await ctx.project(SESSION)
        n = e.data["turn"]
        if n >= len(st.turns) or st.turns[n].status != "pending":
            return []  # a redelivery after the reply was recorded
        sid = e.run_id or ""
        done = SessionState(turns=st.turns[:n], summaries=st.summaries)
        history = history_messages(done, self.system)
        messages = [*history, Message(role="user", content=st.turns[n].user)]
        out: list[Event] = []
        hint: dict[str, Any] = {}
        compacted = False
        if self.context is not None:
            from aaa.context import SummaryBook, SummaryRecord

            book = SummaryBook([SummaryRecord(s.upto, s.prefix_hash, s.text) for s in st.summaries])
            prep = await self.context.prepare(messages, book=book)
            if prep.report is not None:
                compacted, messages = True, prep.messages
            for i, rec in enumerate(prep.new_summaries):
                spend = (
                    {"usage": {"total_tokens": prep.tokens_spent}, "cost": prep.cost}
                    if i == 0
                    else {}
                )
                out.append(
                    e.caused(
                        SUMMARY,
                        {
                            "upto": rec.upto,
                            "prefix_hash": rec.prefix_hash,
                            "text": rec.text,
                            **spend,
                        },
                        id=f"{sid}:summary:{rec.upto}",
                    )
                )
        last = next((t for t in reversed(done.turns) if t.status == "done"), None)
        if not compacted and last is not None and last.state_id:
            # the provider already holds everything before the new message (the LLMHandler checks it can use that)
            hint["provider_state"] = {
                "state_id": last.state_id,
                "covered": sum(m.role != "system" for m in history),
            }
        started = await ctx.started()
        await ctx.engine.start(
            self.inner,
            {"messages": [m.model_dump(exclude_none=True) for m in messages], **hint},
            run_id=f"{sid}:t{n}",
            parent=sid,
            tag=f"turn:{n}",
            budget=self.turn_budget,
            meta=dict(started.meta),
        )
        return out

    async def _record_reply(self, e: Event, ctx: Ctx) -> list[Event]:
        tag = e.data.get("tag") or ""
        if not tag.startswith("turn:"):
            return []
        n, child = int(tag.removeprefix("turn:")), e.data["child_run_id"]
        outcome = e.data.get("outcome") or {}
        value = outcome.get("value")
        text = (
            value.get("text")
            if isinstance(value, dict)
            else (value if isinstance(value, str) else "")
        )
        state_id = None
        for ev in reversed(await ctx.engine.retry(lambda: ctx.engine.store.read(child), "read")):
            if (
                ev.type == "aaa.llm.completed"
                and ev.data.get("tag") in (None, "main")
                and ev.data.get("state_id")
            ):
                state_id = ev.data["state_id"]
                break
        data = {
            "turn": n,
            "text": text or "",
            "kind": outcome.get("kind"),
            "reason": outcome.get("reason"),
            "run_id": child,
            **({"state_id": state_id} if state_id else {}),
        }
        return [e.caused(REPLY, data, id=f"{e.run_id}:reply:{n}")]


# ---- client side ------------------------------------------------------------------------------
@dataclass
class TurnResult:
    turn: int
    text: str
    kind: str
    reason: str | None = None
    run_id: str | None = None

    @property
    def ok(self) -> bool:
        return self.kind == OutcomeKind.SUCCESS.value


@dataclass
class SessionInfo:
    id: str
    tenant: str | None
    user: str | None
    open: bool


class TurnStream:
    """`async for piece in chat.stream(...)` yields the reply as it is written; `await s.result()` is the final turn."""

    def __init__(self, session: Session, turn: int, seq: int) -> None:
        self.session, self.turn, self._seq = session, turn, seq

    def __aiter__(self) -> AsyncIterator[str]:
        return self._pieces()

    async def _pieces(self) -> AsyncIterator[str]:
        eng = self.session.engine
        async for e in eng.store.subscribe(f"{self.session.id}:t{self.turn}"):
            if e.type == "aaa.llm.delta" and e.data.get("tag") in (None, "main"):
                yield e.data["text"]
            elif e.type == RUN_COMPLETED:
                return

    async def result(self) -> TurnResult:
        return await self.session.wait(self.turn, _from_seq=self._seq)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None


class Session:
    def __init__(self, manager: SessionManager, session_id: str) -> None:
        self.manager, self.id = manager, session_id
        self.engine = manager.engine

    async def _state(self) -> SessionState:
        return await self.engine.project(SESSION, self.id)

    async def history(self) -> list[Turn]:
        return list((await self._state()).turns)

    async def messages(self) -> list[Message]:
        return history_messages(await self._state(), self.manager.system)

    async def _closed_reason(self) -> str | None:
        meta = await self.engine.project(RUNMETA, self.id)
        if meta.completed is None:
            return None
        return meta.completed.get("reason") or meta.completed.get("kind") or "closed"

    async def _submit(self, text: str) -> tuple[int, int]:
        if (why := await self._closed_reason()) is not None:
            raise SessionClosed(f"session {self.id} is closed ({why})")
        st = await self._state()
        if any(t.status == "pending" for t in st.turns):
            raise SessionBusy(f"session {self.id} is still answering turn {len(st.turns) - 1}")
        n, nonce = len(st.turns), new_id()
        stored = await self.engine.emit(
            Event(
                id=f"{self.id}:turn:{n}",
                type=MESSAGE,
                stream=self.id,
                run_id=self.id,
                source="session",
                data={"turn": n, "text": text, "nonce": nonce},
            )
        )
        if stored is None:
            raise SessionClosed(f"session {self.id} is closed")
        if stored.dup and stored.data.get("nonce") != nonce:  # someone else claimed this turn first
            raise SessionBusy(f"session {self.id}: turn {n} was taken by another sender")
        return n, stored.seq

    async def wait(
        self, turn: int, *, timeout_s: float | None = None, _from_seq: int = 0
    ) -> TurnResult:
        """The reply to `turn`, from any process, even if the one that sent it died. Raises `SessionClosed` if
        the session ends first."""

        def match(e: Event) -> bool:
            return (e.type == REPLY and e.data.get("turn") == turn) or e.type == RUN_COMPLETED

        ev = await wait_for(self.engine.await_event(self.id, match, from_seq=_from_seq), timeout_s)
        if ev.type == RUN_COMPLETED:
            # The session ended first (closed, or this very turn used up its budget). If the turn's own run
            # finished, its answer was paid for: hand it over instead of losing it.
            child = f"{self.id}:t{turn}"
            if await self.engine.retry(lambda: self.engine.store.last_seq(child), "read") >= 0:
                out = await wait_for(self.engine.handle(child).result(), 10)
                if out.ok:
                    value = out.value
                    reply = (
                        value.get("text")
                        if isinstance(value, dict)
                        else (value if isinstance(value, str) else "")
                    )
                    why = f"session ended: {ev.data.get('reason')}"
                    return TurnResult(turn, reply or "", out.kind.value, why, child)
            raise SessionClosed(
                f"session {self.id} ended before turn {turn} was answered ({ev.data.get('reason')})"
            )
        d = ev.data
        return TurnResult(
            turn, d.get("text") or "", d.get("kind") or "failed", d.get("reason"), d.get("run_id")
        )

    async def send(self, text: str, *, timeout_s: float | None = None) -> TurnResult:
        """Say something and wait for the answer. One turn at a time (`SessionBusy`)."""
        n, seq = await self._submit(text)
        return await self.wait(n, timeout_s=timeout_s, _from_seq=seq)

    async def stream(self, text: str) -> TurnStream:
        """Like `send`, but iterate the reply as it is written (needs an inner algorithm built with `stream=True`)."""
        n, seq = await self._submit(text)
        return TurnStream(self, n, seq)

    async def close(self, reason: str = "closed") -> None:
        """End the session. A running turn is cancelled. A closed session is an ordinary finished run, so it can
        be erased and falls under retention like any other."""
        st = await self._state()
        await self.engine.complete(
            self.id, OutcomeKind.SUCCESS, value={"turns": len(st.turns)}, reason=reason
        )


class SessionManager:
    """Creates, finds and lists sessions for one inner algorithm. Build it in EVERY process that may handle a
    turn (workers register the algorithms; clients just call it)."""

    def __init__(
        self,
        engine: Engine,
        algo: Algorithm,
        *,
        system: str = "",
        context: ContextPolicy | None = None,
        turn_budget: Budget | None = None,
        session_budget: Budget | None = None,
    ) -> None:
        self.engine, self.inner, self.system = engine, algo, system
        self.session_budget = session_budget
        self.algo = SessionAlgo(algo, system=system, context=context, turn_budget=turn_budget)
        engine.register(self.algo)
        engine.register(algo)

    async def create(
        self,
        *,
        tenant: str | None = None,
        user: str | None = None,
        title: str | None = None,
        meta: dict[str, str | int | float | bool] | None = None,
        session_id: str | None = None,
    ) -> Session:
        """A new session. `user` is who it is for (memory is scoped by it; it is never taken from the model)."""
        sid = session_id or f"s-{new_id()}"
        if tenant is not None:
            from aaa.runtime.tenancy import TenantScope

            sid = TenantScope(self.engine, tenant).run_id(sid)
        info: dict[str, str | int | float | bool] = {
            **(meta or {}),
            **({"user": user} if user else {}),
        }
        await self.engine.start(
            self.algo,
            {"inner": self.inner.name, **({"title": title} if title else {})},
            run_id=sid,
            tenant=tenant,
            meta=info,
            budget=self.session_budget,
        )
        return Session(self, sid)

    async def get(self, session_id: str, *, tenant: str | None = None) -> Session:
        """Reattach to an existing session. `KeyError` if it does not exist or belongs to another tenant."""
        try:
            started = await self.engine._started_of(session_id)
        except KeyError:
            raise KeyError(f"no session {session_id!r}") from None
        if started.data.get("algo") != self.algo.name or (
            tenant is not None and started.tenant != tenant
        ):
            raise KeyError(f"no session {session_id!r}")
        return Session(self, session_id)

    async def list(
        self, *, tenant: str | None = None, user: str | None = None, open_only: bool = True
    ) -> list[SessionInfo]:
        out: list[SessionInfo] = []
        for r in await self.engine.retry(self.engine.store.run_index, "read"):
            s = r.started
            if s.data.get("algo") != self.algo.name:
                continue
            if (tenant is not None and s.tenant != tenant) or (
                user is not None and s.meta.get("user") != user
            ):
                continue
            if open_only and r.completed is not None:
                continue
            u = s.meta.get("user")
            out.append(
                SessionInfo(
                    r.stream, s.tenant, str(u) if u is not None else None, r.completed is None
                )
            )
        return out

    def tenant(self, name: str) -> TenantSessions:
        return TenantSessions(self, name)


class TenantSessions:
    """Sessions of one tenant: ids are namespaced and a lookup of another tenant's session is a plain `KeyError`."""

    def __init__(self, manager: SessionManager, tenant: str) -> None:
        from aaa.runtime.tenancy import valid_tenant

        self.manager, self.tenant = manager, valid_tenant(tenant)

    async def create(self, **kw: Any) -> Session:
        return await self.manager.create(tenant=self.tenant, **kw)

    async def get(self, session_id: str) -> Session:
        from aaa.runtime.tenancy import TenantScope

        sid = TenantScope(self.manager.engine, self.tenant).run_id(session_id)
        return await self.manager.get(sid, tenant=self.tenant)

    async def list(self, **kw: Any) -> list[SessionInfo]:
        return await self.manager.list(tenant=self.tenant, **kw)


__all__ = [
    "SESSION",
    "Session",
    "SessionAlgo",
    "SessionBusy",
    "SessionClosed",
    "SessionInfo",
    "SessionManager",
    "SessionState",
    "TenantSessions",
    "Turn",
    "TurnResult",
    "TurnStream",
    "history_messages",
]
