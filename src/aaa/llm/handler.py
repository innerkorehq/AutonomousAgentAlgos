from __future__ import annotations

from aaa.events import Event
from aaa.llm.client import LLMClient
from aaa.llm.errors import LLMError
from aaa.llm.streaming import collect
from aaa.llm.types import LLMRequest, LLMResponse
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

    async def handle(self, e: Event) -> list[Event]:
        tag = e.data.get("tag")
        try:
            req = LLMRequest.model_validate(e.data["request"])
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
