from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from aaa.algos.base import finish, llm_request, parse_json_obj, strip_fence, task_messages
from aaa.core.outcome import OutcomeKind
from aaa.events import Event, Projection
from aaa.llm.types import LLMRequest, LLMResponse, Message
from aaa.runtime import Ctx

_CRITIC_SCHEMA = {
    "type": "object",
    "properties": {"ok": {"type": "boolean"}, "feedback": {"type": "string"}},
    "required": ["ok", "feedback"],
    "additionalProperties": False,
}
_CRITIC_SYSTEM = (
    "You are a strict reviewer. Judge whether the candidate fully satisfies the task. "
    'Reply with JSON {"ok": bool, "feedback": "specific, actionable fixes"}.'
)


class _S(BaseModel):
    input: dict[str, Any] = Field(default_factory=dict)
    request_id: str | None = None  # last `gen` request: a revision continues from it
    iterations: int = 0
    candidate: str = ""


def _apply(st: _S, e: Event) -> _S:
    d = e.data
    if e.type == "aaa.run.started":
        st.input = d["input"]
    elif e.type == "aaa.llm.requested" and d.get("tag") == "gen":
        st.request_id = e.id
    elif e.type == "aaa.llm.completed" and d.get("tag") == "gen":
        msg = LLMResponse.model_validate(d["response"]).message
        st.candidate, st.iterations = msg.content or "", st.iterations + 1
    return st


REFINE = Projection(_S, _apply, name="aaa.refine", version=2, state_type=_S)


class _Refine:
    """generate -> verify -> (feedback) -> regenerate, bounded by max_iterations. The verifier
    is injected by subclasses; the generator keeps the whole attempt/feedback history."""

    name = "refine"

    def __init__(
        self,
        *,
        system: str = "",
        model: str | None = None,
        max_iterations: int = 3,
        params: dict[str, Any] | None = None,
        extract_code: bool = True,
    ) -> None:
        self.system, self.model, self.max_iterations = system, model, max_iterations
        self.params, self.extract = params or {}, extract_code

    def subscriptions(self) -> list[str]:
        return ["aaa.run.started", "aaa.llm.completed", "aaa.llm.failed", "aaa.validate.completed"]

    def _gen(self, e: Event, messages: list[Message], prev: str | None = None) -> Event:
        """`prev` set => `messages` are only the NEW ones, continuing that request's conversation."""
        req = LLMRequest(model=self.model, messages=messages, params=self.params)
        return llm_request(e, req, "gen", prev=prev)

    def verify(self, e: Event, st: _S, candidate: str) -> Event:
        raise NotImplementedError

    async def on_event(self, e: Event, ctx: Ctx) -> list[Event]:
        tag = e.data.get("tag")
        st = await ctx.project(REFINE)
        if e.type == "aaa.run.started":
            return [self._gen(e, task_messages(st.input, self.system))]
        if e.type == "aaa.llm.failed":
            err = e.data["error"]
            return [finish(e, OutcomeKind.FAILED, reason=f"{err['type']}: {err['message']}")]
        if e.type == "aaa.llm.completed" and tag == "gen":
            cand = strip_fence(st.candidate) if self.extract else st.candidate
            return [self.verify(e, st, cand)]
        # verdict from validator or critic
        if e.type == "aaa.validate.completed":
            ok, fb = e.data["verdict"]["ok"], e.data["verdict"]["feedback"]
        elif e.type == "aaa.llm.completed" and tag == "critic":
            obj = parse_json_obj(LLMResponse.model_validate(e.data["response"]).message.content)
            ok, fb = (
                (bool(obj.get("ok")), str(obj.get("feedback", "")))
                if obj
                else (False, "critic returned invalid JSON")
            )
        else:
            return []
        cand = strip_fence(st.candidate) if self.extract else st.candidate
        if ok:
            return [
                finish(e, OutcomeKind.SUCCESS, {"candidate": cand, "iterations": st.iterations})
            ]
        if st.iterations >= self.max_iterations:
            return [
                finish(
                    e,
                    OutcomeKind.NO_SOLUTION,
                    {"candidate": cand, "iterations": st.iterations},
                    reason=f"not accepted after {st.iterations} attempts: {fb}",
                )
            ]
        fix = Message(
            role="user",
            content=f"Your answer was rejected.\nFeedback: {fb}\nProduce a corrected, complete answer.",
        )
        return [self._gen(e, [fix], prev=st.request_id)]


class GenerateTestRepair(_Refine):
    """Candidate is checked by a deterministic named validator (tests, linter, schema, rules)
    served by ValidationHandler. Accepts only what the validator accepts."""

    name = "generate_test_repair"

    def __init__(self, validator: str = "default", **kw: Any) -> None:
        super().__init__(**kw)
        self.validator = validator

    def verify(self, e: Event, st: _S, candidate: str) -> Event:
        return e.caused(
            "aaa.validate.requested",
            {
                "candidate": candidate,
                "validator": self.validator,
                "context": {"task": st.input, "iteration": st.iterations},
            },
        )


class GeneratorCritic(_Refine):
    """Candidate is judged by a separate LLM critic call. Weaker than deterministic checks
    (the critic is probabilistic); always bounded by max_iterations."""

    name = "generator_critic"

    def __init__(
        self, critic_system: str = _CRITIC_SYSTEM, critic_model: str | None = None, **kw: Any
    ) -> None:
        super().__init__(**kw)
        self.critic_system, self.critic_model = critic_system, critic_model

    def verify(self, e: Event, st: _S, candidate: str) -> Event:
        msgs = [
            Message(role="system", content=self.critic_system),
            Message(
                role="user", content=f"Task:\n{st.input.get('task', '')}\n\nCandidate:\n{candidate}"
            ),
        ]
        req = LLMRequest(
            model=self.critic_model or self.model, messages=msgs, response_schema=_CRITIC_SCHEMA
        )
        return llm_request(e, req, "critic")
