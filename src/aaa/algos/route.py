from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from aaa.algos.base import finish, llm_failed, llm_request, parse_json_obj
from aaa.core.ids import stable_id
from aaa.core.outcome import OutcomeKind
from aaa.events import Event
from aaa.llm.types import LLMRequest, LLMResponse, Message
from aaa.runtime import Algorithm, Ctx


@dataclass
class Route:
    algo: Algorithm
    description: str = ""


class Router:
    """Pick exactly one specialist and delegate (as a child run). Deterministic `classifier`
    first (cheap, testable); the LLM only decides what rules don't. Returns the child's outcome."""

    name = "router"

    def __init__(
        self,
        routes: dict[str, Route | Algorithm],
        *,
        classifier: Callable[[dict[str, Any]], str | None] | None = None,
        default: str | None = None,
        model: str | None = None,
    ) -> None:
        self.routes = {k: v if isinstance(v, Route) else Route(v) for k, v in routes.items()}
        self.classifier, self.default, self.model = classifier, default, model

    def subscriptions(self) -> list[str]:
        return ["aaa.run.started", "aaa.llm.completed", "aaa.llm.failed", "aaa.child.completed"]

    async def _delegate(self, e: Event, ctx: Ctx, route: str, inp: dict[str, Any]) -> list[Event]:
        await ctx.engine.start(
            self.routes[route].algo,
            inp,
            parent=ctx.run_id,
            tag=route,
            run_id=stable_id(ctx.run_id, route),
        )
        return [e.caused("aaa.route.selected", {"route": route})]

    async def on_event(self, e: Event, ctx: Ctx) -> list[Event]:
        if e.type == "aaa.child.completed":
            o = e.data["outcome"]
            return [finish(e, OutcomeKind(o["kind"]), o["value"], o.get("reason"))]
        if e.type == "aaa.llm.failed":
            return [llm_failed(e)]
        inp = (await ctx.started()).data["input"]
        if e.type == "aaa.llm.completed":
            obj = (
                parse_json_obj(LLMResponse.model_validate(e.data["response"]).message.content) or {}
            )
            name = obj.get("route") if obj.get("route") in self.routes else self.default
            if name is None:
                return [
                    finish(
                        e,
                        OutcomeKind.INVALID_INPUT,
                        reason=f"router chose unknown route {obj.get('route')!r}",
                    )
                ]
            return await self._delegate(e, ctx, name, inp)
        name = self.classifier(inp) if self.classifier else None
        if name in self.routes:
            return await self._delegate(e, ctx, name, inp)
        menu = "\n".join(f"- {k}: {r.description}" for k, r in self.routes.items())
        schema = {
            "type": "object",
            "properties": {"route": {"type": "string", "enum": list(self.routes)}},
            "required": ["route"],
            "additionalProperties": False,
        }
        req = LLMRequest(
            model=self.model,
            response_schema=schema,
            messages=[
                Message(
                    role="system",
                    content=f'Choose the best handler for the request. Reply JSON {{"route": name}}.\n{menu}',
                ),
                Message(role="user", content=str(inp.get("task", ""))),
            ],
        )
        return [llm_request(e, req, "route")]
