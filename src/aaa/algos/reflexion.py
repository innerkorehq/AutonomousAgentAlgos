"""Reflexion: learn from failed attempts in words, then try again from scratch.

Unlike `GeneratorCritic` (which keeps one conversation and patches the answer), every trial here is a FRESH attempt whose prompt
carries the short lessons ("reflections") written after the earlier failures. The reflections are plain events in the run's log
and in the result, so you can store them (for example with a memory backend) and pass them back in as `input["reflections"]` to
start a later run already knowing them."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from aaa.algos.base import (
    finish,
    llm_failed,
    llm_request,
    parse_json_obj,
    strip_fence,
    task_messages,
)
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
    "You are a strict evaluator. Judge whether the attempt fully solves the task. "
    'Reply with JSON {"ok": bool, "feedback": "what is wrong, specifically"}.'
)
_REFLECT_SYSTEM = (
    "You are reviewing a failed attempt. In at most three sentences, state what went wrong and what to do "
    "differently next time. Write advice for your future self, not a new answer."
)
MAX_REFLECTION_CHARS = 600


class _S(BaseModel):
    input: dict[str, Any] = Field(default_factory=dict)
    trial: int = 0  # attempts started
    attempt: str = ""  # the latest attempt's text
    feedback: str = ""  # the evaluator's verdict on it
    reflections: list[str] = Field(default_factory=list)
    attempts_done: int = 0


def _apply(st: _S, e: Event) -> _S:
    d = e.data
    if e.type == "aaa.run.started":
        st.input = d["input"]
        st.reflections = [str(r)[:MAX_REFLECTION_CHARS] for r in d["input"].get("reflections", [])]
    elif e.type == "aaa.llm.requested" and d.get("tag") == "attempt":
        st.trial += 1
    elif e.type == "aaa.llm.completed" and d.get("tag") == "attempt":
        st.attempt = LLMResponse.model_validate(d["response"]).message.content or ""
        st.attempts_done += 1
    elif e.type == "aaa.reflexion.feedback":
        st.feedback = d["feedback"]
    elif e.type == "aaa.llm.completed" and d.get("tag") == "reflect":
        text = (LLMResponse.model_validate(d["response"]).message.content or "").strip()
        if text:
            st.reflections.append(text[:MAX_REFLECTION_CHARS])
    return st


REFLEXION = Projection(_S, _apply, name="aaa.reflexion", version=1, state_type=_S)


class Reflexion:
    """Trial -> evaluate -> (if rejected) reflect -> new trial, at most `max_trials` times.

    The evaluator is a deterministic named validator (`validator="..."`, served by `ValidationHandler`) or, if none is given,
    an LLM critic. `keep_reflections` bounds how many lessons enter the prompt (the most recent ones).
    Result: `{"candidate", "trials", "reflections"}`; a run that never passes ends `NO_SOLUTION` carrying the best-effort
    last attempt and the reflections."""

    name = "reflexion"

    def __init__(
        self,
        *,
        validator: str | None = None,
        system: str = "",
        model: str | None = None,
        critic_model: str | None = None,
        critic_system: str = _CRITIC_SYSTEM,
        max_trials: int = 3,
        keep_reflections: int = 5,
        params: dict[str, Any] | None = None,
        extract_code: bool = False,
    ) -> None:
        if not 1 <= max_trials <= 10:
            raise ValueError("max_trials must be between 1 and 10")
        self.validator, self.system, self.model = validator, system, model
        self.critic_model, self.critic_system = critic_model, critic_system
        self.max_trials, self.keep, self.params = max_trials, keep_reflections, params or {}
        self.extract = extract_code

    def subscriptions(self) -> list[str]:
        return ["aaa.run.started", "aaa.llm.completed", "aaa.llm.failed", "aaa.validate.completed"]

    # -- prompts ------------------------------------------------------------------------------
    def _attempt(self, e: Event, st: _S) -> Event:
        msgs = task_messages(st.input, self.system)
        if st.reflections:
            lessons = "\n".join(f"- {r}" for r in st.reflections[-self.keep :])
            msgs.insert(
                len([m for m in msgs if m.role == "system"]),
                Message(
                    role="system",
                    content=f"Lessons from earlier failed attempts at this task:\n{lessons}\nApply them.",
                ),
            )
        return llm_request(
            e, LLMRequest(model=self.model, messages=msgs, params=self.params), "attempt"
        )

    def _evaluate(self, e: Event, st: _S, candidate: str) -> Event:
        if self.validator is not None:
            return e.caused(
                "aaa.validate.requested",
                {
                    "candidate": candidate,
                    "validator": self.validator,
                    "context": {"task": st.input, "trial": st.trial},
                },
            )
        msgs = [
            Message(role="system", content=self.critic_system),
            Message(
                role="user", content=f"Task:\n{st.input.get('task', '')}\n\nAttempt:\n{candidate}"
            ),
        ]
        req = LLMRequest(
            model=self.critic_model or self.model, messages=msgs, response_schema=_CRITIC_SCHEMA
        )
        return llm_request(e, req, "critic")

    def _reflect(self, e: Event, st: _S, candidate: str) -> Event:
        msgs = [
            Message(role="system", content=_REFLECT_SYSTEM),
            Message(
                role="user",
                content=(
                    f"Task:\n{st.input.get('task', '')}\n\nAttempt:\n{candidate}\n\n"
                    f"Evaluator feedback:\n{st.feedback}"
                ),
            ),
        ]
        return llm_request(
            e, LLMRequest(model=self.critic_model or self.model, messages=msgs), "reflect"
        )

    # -- driver -------------------------------------------------------------------------------
    async def on_event(self, e: Event, ctx: Ctx) -> list[Event]:
        tag = e.data.get("tag")
        if e.type == "aaa.run.started":
            return [self._attempt(e, await ctx.project(REFLEXION))]
        if e.type == "aaa.llm.failed":
            return [llm_failed(e)]
        st = await ctx.project(REFLEXION)
        candidate = strip_fence(st.attempt) if self.extract else st.attempt
        if e.type == "aaa.llm.completed" and tag == "attempt":
            return [self._evaluate(e, st, candidate)]
        if e.type == "aaa.validate.completed":
            ok, fb = bool(e.data["verdict"]["ok"]), str(e.data["verdict"]["feedback"])
        elif e.type == "aaa.llm.completed" and tag == "critic":
            obj = parse_json_obj(LLMResponse.model_validate(e.data["response"]).message.content)
            ok, fb = (
                (bool(obj.get("ok")), str(obj.get("feedback", "")))
                if obj
                else (False, "the evaluator returned invalid JSON")
            )
        elif e.type == "aaa.llm.completed" and tag == "reflect":
            return [self._attempt(e, st)]  # the lesson is in the state now: a fresh attempt
        else:
            return []
        if ok:
            return [
                finish(
                    e,
                    OutcomeKind.SUCCESS,
                    {"candidate": candidate, "trials": st.trial, "reflections": st.reflections},
                )
            ]
        if st.trial >= self.max_trials:
            return [
                finish(
                    e,
                    OutcomeKind.NO_SOLUTION,
                    {"candidate": candidate, "trials": st.trial, "reflections": st.reflections},
                    reason=f"not accepted after {st.trial} trials: {fb}",
                )
            ]
        record = e.caused("aaa.reflexion.feedback", {"trial": st.trial, "feedback": fb})
        st.feedback = fb  # the reflect prompt needs it now; the event makes it durable
        return [record, self._reflect(record, st, candidate)]
