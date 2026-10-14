from __future__ import annotations

from collections import OrderedDict

from aaa.events import Event
from aaa.llm.client import LLMClient
from aaa.llm.errors import FatalError, LLMError
from aaa.llm.streaming import collect
from aaa.llm.types import LLMRequest, LLMResponse, Message
from aaa.runtime import Engine

REQUESTED = "aaa.llm.requested"
COMPLETED = "aaa.llm.completed"
FAILED = "aaa.llm.failed"
DELTA = "aaa.llm.delta"


def _result(request: Event, type: str, data: dict) -> Event:
    """One result per request on the log, whichever worker finishes first (deterministic id)."""
    return request.caused(type, data, id=f"{request.id}:result")


class LLMHandler:
    """Capability handler: `aaa.llm.requested` in -> `aaa.llm.completed|failed` out.
    Request data: {"request": LLMRequest, "tag": optional str echoed back}. Failures are facts
    on the log (the algorithm decides what to do), never exceptions. The completed event
    carries top-level `usage`/`cost` so the Engine's budget ledger sees them. With
    `request.stream` and a streaming client, `aaa.llm.delta` events are emitted as text arrives.
    Tool-call ids are rewritten to be unique per call (providers may omit or repeat them)."""

    def __init__(
        self, client: LLMClient, *, concurrency: int = 8, emit_deltas: bool = True
    ) -> None:
        self.client, self.concurrency, self.emit_deltas = client, concurrency, emit_deltas
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
                    await self.engine.emit(
                        e.caused(
                            DELTA, {"text": ch.text, "tag": e.data.get("tag")}, id=f"{e.id}:d{n}"
                        )
                    )
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
        for ev in await self.engine.store.read(stream):
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

    async def handle(self, e: Event) -> list[Event]:
        tag = e.data.get("tag")
        try:
            req = LLMRequest.model_validate(e.data["request"])
            req.messages = await self._transcript(e, req)
            resp = await self._run(e, req)
        except LLMError as exc:
            return [
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
                )
            ]
        for i, call in enumerate(resp.message.tool_calls):
            call.id = f"call_{e.id}_{i}"
        self._remember(e.id, [*req.messages, resp.message], (e.data.get("conv") or {}).get("prev"))
        return [
            _result(
                e,
                COMPLETED,
                {
                    "tag": tag,
                    "response": resp.model_dump(mode="json"),
                    "usage": resp.usage.model_dump(),
                    "cost": resp.cost,
                },
            )
        ]
