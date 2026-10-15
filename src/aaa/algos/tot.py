"""Tree-of-thoughts as a beam search: propose several next steps, score them, keep the best few, repeat.

Every model call is an event with a tag that names its place in the tree (`p<depth>:<beam>.<branch>` to propose,
`s<depth>:<beam>.<branch>` to score), so the state is a pure fold of the log and the order in which parallel calls finish never
changes the outcome. Cost is bounded at construction: at most `depth * width * branching` proposals plus the same number of scores."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from aaa.algos.base import finish, llm_request, parse_json_obj, stable, task_messages
from aaa.core.outcome import OutcomeKind
from aaa.events import Event, Projection
from aaa.llm.types import LLMRequest, LLMResponse, Message
from aaa.runtime import Ctx

MAX_DEPTH, MAX_FANOUT = 8, 25
FINAL = "FINAL:"
_SCORE_SCHEMA = {
    "type": "object",
    "properties": {"score": {"type": "number"}},
    "required": ["score"],
    "additionalProperties": False,
}


class _S(BaseModel):
    input: dict[str, Any] = Field(default_factory=dict)
    thoughts: dict[str, str] = Field(default_factory=dict)  # "<depth>:<beam>.<branch>" -> text
    scores: dict[str, float] = Field(default_factory=dict)
    bad: list[str] = Field(default_factory=list)  # proposal/score calls that failed ("p..."/"s...")


def _apply(st: _S, e: Event) -> _S:
    tag = str(e.data.get("tag", ""))
    if e.type == "aaa.run.started":
        st.input = e.data["input"]
    elif e.type == "aaa.llm.completed" and tag[:1] == "p":
        st.thoughts[tag[1:]] = (
            LLMResponse.model_validate(e.data["response"]).message.content or ""
        ).strip()
    elif e.type == "aaa.llm.completed" and tag[:1] == "s":
        obj = parse_json_obj(LLMResponse.model_validate(e.data["response"]).message.content)
        try:
            st.scores[tag[1:]] = max(0.0, min(10.0, float((obj or {})["score"])))
        except (KeyError, TypeError, ValueError):
            st.scores[tag[1:]] = 0.0
    elif e.type == "aaa.llm.failed" and tag[:1] in ("p", "s"):
        st.bad.append(tag)
    return st


TOT = Projection(_S, _apply, name="aaa.tot", version=1, state_type=_S)


class TreeOfThoughts:
    """Beam search over reasoning steps.

    At each of up to `depth` levels, every node of the beam gets `branching` proposed next steps; each proposal is scored 0-10 by
    a separate call; the best `width` survive. A proposal that starts with `FINAL:` is a complete answer. The search stops when the
    best node of a level is final, when its score reaches `stop_score`, or at the last level, and returns the best path.

    Result: `{"answer", "score", "path": [steps...], "depth", "explored"}`."""

    name = "tree_of_thoughts"

    def __init__(
        self,
        *,
        depth: int = 3,
        width: int = 2,
        branching: int = 3,
        stop_score: float | None = 9.0,
        model: str | None = None,
        scorer_model: str | None = None,
        system: str = "You solve problems one reasoning step at a time.",
        temperature: float = 0.8,
    ) -> None:
        if not 1 <= depth <= MAX_DEPTH:
            raise ValueError(f"depth must be between 1 and {MAX_DEPTH}")
        if width < 1 or branching < 1 or width * branching > MAX_FANOUT:
            raise ValueError(f"width * branching must be between 1 and {MAX_FANOUT}")
        self.depth, self.width, self.branching = depth, width, branching
        self.stop_score, self.model, self.scorer_model = stop_score, model, scorer_model
        self.system, self.temperature = system, temperature

    @property
    def max_calls(self) -> int:
        """The most model calls one run can make."""
        return self.depth * self.width * self.branching * 2

    def subscriptions(self) -> list[str]:
        return ["aaa.run.started", "aaa.llm.completed", "aaa.llm.failed"]

    # -- the tree, derived from the state ------------------------------------------------------
    def _beam(self, st: _S, d: int) -> list[tuple[str, list[str]]]:
        """The nodes (key, path of steps) that level `d` expands. Level 0 expands the empty root."""
        if d == 0:
            return [("0", [])]
        prev = self._beam(st, d - 1)
        ranked = self._ranked(st, d - 1, prev)
        return [(k, path) for k, path, _ in ranked if not path[-1].startswith(FINAL)][: self.width]

    def _ranked(
        self, st: _S, d: int, beam: list[tuple[str, list[str]]]
    ) -> list[tuple[str, list[str], float]]:
        """Level `d`'s scored candidates, best first (ties broken by key, so the order is deterministic)."""
        out = []
        for b, (_, path) in enumerate(beam):
            for k in range(self.branching):
                key = f"{d}:{b}.{k}"
                if key in st.thoughts and st.thoughts[key] and key in st.scores:
                    out.append((key, [*path, st.thoughts[key]], st.scores[key]))
        return sorted(out, key=lambda c: (-c[2], c[0]))

    # -- prompts -----------------------------------------------------------------------------
    def _propose(self, e: Event, st: _S, d: int, b: int, path: list[str], k: int) -> Event:
        steps = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(path)) or "(none yet)"
        task = task_messages(st.input, "")[-1].content or ""
        req = LLMRequest(
            model=self.model,
            params={"temperature": self.temperature},
            messages=[
                Message(role="system", content=self.system),
                Message(
                    role="user",
                    content=(
                        f"Task:\n{task}\n\nReasoning so far:\n{steps}\n\n"
                        f"Write the next step only. If this step completes the solution, start it with '{FINAL}' "
                        "followed by the final answer."
                    ),
                ),
            ],
        )
        return stable(llm_request(e, req, f"p{d}:{b}.{k}"), f"tot:p{d}:{b}.{k}")

    def _score(self, e: Event, st: _S, d: int, key: str, path: list[str]) -> Event:
        task = task_messages(st.input, "")[-1].content or ""
        steps = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(path))
        req = LLMRequest(
            model=self.scorer_model or self.model,
            messages=[
                Message(
                    role="system",
                    content='Rate how promising this line of reasoning is for solving the task, from 0 (wrong or stuck) to 10 (solved). Reply with JSON {"score": number}.',
                ),
                Message(role="user", content=f"Task:\n{task}\n\nReasoning:\n{steps}"),
            ],
            response_schema=_SCORE_SCHEMA,
        )
        return stable(llm_request(e, req, f"s{key}"), f"tot:s{key}")

    # -- driver ------------------------------------------------------------------------------
    def _expand(self, e: Event, st: _S, d: int) -> list[Event]:
        return [
            self._propose(e, st, d, b, path, k)
            for b, (_, path) in enumerate(self._beam(st, d))
            for k in range(self.branching)
        ]

    async def on_event(self, e: Event, ctx: Ctx) -> list[Event]:
        if e.type == "aaa.run.started":
            return self._expand(e, await ctx.project(TOT), 0)
        tag = str(e.data.get("tag", ""))
        if tag[:1] not in ("p", "s"):
            return []
        st = await ctx.project(TOT)
        d = int(tag[1:].split(":")[0])
        beam = self._beam(st, d)
        keys = [f"{d}:{b}.{k}" for b in range(len(beam)) for k in range(self.branching)]
        if not all(k in st.thoughts or f"p{k}" in st.bad for k in keys):
            return []  # proposals of this level still coming in
        cands = [
            (f"{d}:{b}.{k}", [*path, st.thoughts[f"{d}:{b}.{k}"]])
            for b, (_, path) in enumerate(beam)
            for k in range(self.branching)
            if st.thoughts.get(f"{d}:{b}.{k}")
        ]
        if not cands:
            return [finish(e, OutcomeKind.FAILED, reason=f"no usable proposals at depth {d + 1}")]
        if (
            tag[0] == "p"
        ):  # the last proposal is in: ask for every score (repeats are dropped by their fixed ids)
            return [self._score(e, st, d, key, path) for key, path in cands]
        if not all(key in st.scores or f"s{key}" in st.bad for key, _ in cands):
            return []  # scores still coming in
        ranked = self._ranked(st, d, beam)
        if not ranked:
            return [finish(e, OutcomeKind.FAILED, reason=f"every score failed at depth {d + 1}")]
        _, path, score = ranked[0]
        done = (
            path[-1].startswith(FINAL)
            or d + 1 >= self.depth
            or (self.stop_score is not None and score >= self.stop_score)
        )
        if not done:
            return self._expand(e, st, d + 1)
        return [
            finish(
                e,
                OutcomeKind.SUCCESS,
                {
                    "answer": path[-1].removeprefix(FINAL).strip(),
                    "score": score,
                    "path": path,
                    "depth": len(path),
                    "explored": len(st.thoughts),
                },
            )
        ]
